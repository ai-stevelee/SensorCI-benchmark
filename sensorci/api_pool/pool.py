"""APIKeyPool: 100-key concurrent dispatcher with rate limiting and retry.

Supports OpenAI, Anthropic, Google (Gemini), and (optionally) open-model
providers (Together AI, HuggingFace) for the dual-track leaderboard.

Loads keys from .env (multi-key supported, comma-separated).
"""

from __future__ import annotations
import os
import time
import threading
import random
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Any
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env — search cwd, then upward from this file's location
# pool.py lives at: 002_SensorCI/code/sensorci/api_pool/pool.py
# .env lives at:    002_SensorCI/.env  (4 levels up)
_candidates = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent.parent.parent / ".env",  # 002_SensorCI/
    Path(__file__).parent.parent.parent / ".env",         # code/
]
_env_path = next((p for p in _candidates if p.exists()), _candidates[0])
load_dotenv(_env_path, override=False)

# API hub proxy base URL — HUB_BASE_URL == AZURE_OPENAI_BASE_URL (same api-hub internal API)
_HUB_BASE_URL = os.environ.get("HUB_BASE_URL", os.environ.get("AZURE_OPENAI_BASE_URL", "")).rstrip("/")


def _load_numbered_keys(prefix: str = "AZURE_OPENAI_API_KEY", max_n: int = 200) -> list[str]:
    """Load numbered env vars like PREFIX_1 … PREFIX_N into a list."""
    return [
        v for i in range(1, max_n + 1)
        if (v := os.environ.get(f"{prefix}_{i}", "").strip())
    ]


# -----------------------------------------------------------------------------
# Provider registry
# -----------------------------------------------------------------------------

@dataclass
class ProviderSpec:
    name: str
    env_var: str
    default_rpm: int
    canonical_models: list[str]


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai", "OPENAI_API_KEYS", 10000,
        ["gpt-5-pro", "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4.1"],
    ),
    "anthropic": ProviderSpec(
        "anthropic", "ANTHROPIC_API_KEYS", 4000,
        ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
    ),
    "google": ProviderSpec(
        "google", "GOOGLE_API_KEYS", 2000,
        ["gemini-3-pro", "gemini-3-flash", "gemini-2.5-pro"],
    ),
    "together": ProviderSpec(
        "together", "TOGETHER_API_KEYS", 600,
        ["meta-llama/llama-4-behemoth", "mistralai/Mistral-Large-2026", "Qwen/Qwen2.5-72B-Instruct"],
    ),
}


def _model_to_provider(model: str) -> str:
    """Heuristic mapping from model name to provider."""
    m = model.lower()
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    if "llama" in m or "mistral" in m or "qwen" in m:
        return "together"
    raise ValueError(f"Unknown model {model!r} — add to _model_to_provider")


# -----------------------------------------------------------------------------
# Rate limiter (token bucket per provider)
# -----------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = threading.Lock()
        self._tokens = float(rpm)
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                # Refill at rpm/60 tokens per second
                self._tokens = min(self.rpm, self._tokens + elapsed * (self.rpm / 60.0))
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Sleep proportional to deficit
                wait = (1.0 - self._tokens) * 60.0 / self.rpm
            time.sleep(min(wait, 1.0))


# -----------------------------------------------------------------------------
# Response dataclass
# -----------------------------------------------------------------------------

@dataclass
class LLMResponse:
    model: str
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    n_attempts: int = 1
    elapsed_s: float = 0.0


# -----------------------------------------------------------------------------
# Pool
# -----------------------------------------------------------------------------

class APIKeyPool:
    """Pool of API keys across multiple providers with rate limiting and retry.

    Usage:
      pool = APIKeyPool()
      resp = pool.call("gpt-5-pro", messages=[{"role":"user","content":"hi"}])

    Concurrent batch:
      results = pool.batch_call("claude-opus-4-7", prompts=[...], max_workers=64)
    """

    def __init__(
        self,
        providers: Iterable[str] = ("openai", "anthropic", "google"),
        max_retries: int = 5,
        backoff_base: float = 1.5,
        request_timeout_s: int = 120,
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.request_timeout_s = request_timeout_s

        self.keys: dict[str, list[str]] = {}
        self.rate_limiters: dict[str, RateLimiter] = {}
        self._key_idx: dict[str, int] = {}
        self._lock = threading.Lock()
        self._cost_total_usd: float = 0.0
        self._cost_lock = threading.Lock()
        self._max_cost_usd = float(os.environ.get("MAX_API_USD", 200))

        # Numbered Azure keys shared by all API hub-backed providers
        _hub_keys: list[str] | None = None

        for p in providers:
            spec = PROVIDERS[p]
            keys_str = os.environ.get(spec.env_var, "").strip()
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]

            # Fall back to HUB_API_KEYS, then numbered AZURE_OPENAI_API_KEY_* keys
            if not keys:
                hub_csv = os.environ.get("HUB_API_KEYS", "").strip()
                keys = [k.strip() for k in hub_csv.split(",") if k.strip()]
            if not keys:
                if _hub_keys is None:
                    _hub_keys = _load_numbered_keys()
                keys = _hub_keys

            if not keys:
                logger.warning(f"No keys found for provider {p!r} (env var {spec.env_var})")
                continue
            self.keys[p] = keys
            rpm = int(os.environ.get(f"RPM_{p.upper()}", spec.default_rpm))
            self.rate_limiters[p] = RateLimiter(rpm)
            self._key_idx[p] = 0
            logger.info(f"Provider {p}: loaded {len(keys)} keys, rpm={rpm}")

    @contextmanager
    def _acquire_key(self, provider: str):
        with self._lock:
            keys = self.keys[provider]
            key = keys[self._key_idx[provider] % len(keys)]
            self._key_idx[provider] += 1
        self.rate_limiters[provider].acquire()
        yield key

    def _check_budget(self) -> None:
        with self._cost_lock:
            if self._cost_total_usd >= self._max_cost_usd:
                raise RuntimeError(
                    f"Cost ceiling hit: ${self._cost_total_usd:.2f} >= ${self._max_cost_usd:.2f}. "
                    f"Set MAX_API_USD higher in .env to continue."
                )

    def _record_cost(self, usd: float) -> None:
        with self._cost_lock:
            self._cost_total_usd += usd

    @property
    def total_cost_usd(self) -> float:
        return self._cost_total_usd

    # -------------------------------------------------------------------------
    # Provider-specific call wrappers
    # -------------------------------------------------------------------------

    def _call_openai(self, model: str, messages: list[dict], key: str, **kw) -> LLMResponse:
        if _HUB_BASE_URL:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=_HUB_BASE_URL,
                api_key=key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
                timeout=self.request_timeout_s,
            )
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key, timeout=self.request_timeout_s)
        call_kwargs: dict = {"model": model, "messages": messages,
                             "max_completion_tokens": kw.get("max_tokens", 40000)}
        if "temperature" in kw:
            call_kwargs["temperature"] = kw["temperature"]
        if kw.get("response_format") == "json":
            call_kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        r = client.chat.completions.create(**call_kwargs)
        elapsed = time.time() - t0
        text = r.choices[0].message.content or ""
        usd = (r.usage.prompt_tokens * 0.000003 + r.usage.completion_tokens * 0.000015)
        self._record_cost(usd)
        return LLMResponse(model=model, text=text, raw={"usage": r.usage.model_dump()},
                           elapsed_s=elapsed)

    def _call_anthropic(self, model: str, messages: list[dict], key: str, **kw) -> LLMResponse:
        import anthropic as anthropic_sdk
        client_kwargs: dict = {"api_key": key, "timeout": self.request_timeout_s}
        if _HUB_BASE_URL:
            client_kwargs["base_url"] = f"{_HUB_BASE_URL}/claude"
        client = anthropic_sdk.Anthropic(**client_kwargs)

        sys_msg = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            else:
                chat_msgs.append(m)
        call_kwargs: dict = {
            "model": model,
            "max_tokens": kw.get("max_tokens", 40000),
            "messages": chat_msgs,
        }
        if sys_msg:
            call_kwargs["system"] = sys_msg
        if "temperature" in kw:
            call_kwargs["temperature"] = kw["temperature"]
        t0 = time.time()
        r = client.messages.create(**call_kwargs)
        elapsed = time.time() - t0
        text = r.content[0].text if r.content else ""
        usd = (r.usage.input_tokens * 0.000015 + r.usage.output_tokens * 0.000075)
        self._record_cost(usd)
        return LLMResponse(model=model, text=text,
                           raw={"input_tokens": r.usage.input_tokens,
                                "output_tokens": r.usage.output_tokens},
                           elapsed_s=elapsed)

    def _call_google(self, model: str, messages: list[dict], key: str, **kw) -> LLMResponse:
        from google import genai
        from google.genai import types as genai_types
        if _HUB_BASE_URL:
            client = genai.Client(
                api_key=key,
                http_options=genai_types.HttpOptions(base_url=f"{_HUB_BASE_URL}/gemini"),
            )
        else:
            client = genai.Client(api_key=key)

        # System messages prepended to user prompt
        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [m["content"] for m in messages if m["role"] != "system"]
        prompt = "\n".join(sys_parts + user_parts)

        t0 = time.time()
        r = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=kw.get("temperature", 0.7),
                max_output_tokens=kw.get("max_tokens", 40000),
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        elapsed = time.time() - t0
        text = r.text or ""
        usd = 0.0
        if r.usage_metadata:
            total = (r.usage_metadata.prompt_token_count or 0) + (r.usage_metadata.candidates_token_count or 0)
            usd = total * 0.000001
        self._record_cost(usd)
        return LLMResponse(model=model, text=text, raw={}, elapsed_s=elapsed)

    def _call_together(self, model: str, messages: list[dict], key: str, **kw) -> LLMResponse:
        # OpenAI-compatible REST endpoint
        import requests
        t0 = time.time()
        r = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": kw.get("max_tokens", 40000),
                "temperature": kw.get("temperature", 0.0),
            },
            timeout=self.request_timeout_s,
        )
        r.raise_for_status()
        elapsed = time.time() - t0
        d = r.json()
        text = d["choices"][0]["message"]["content"]
        usd = 0.001  # placeholder
        self._record_cost(usd)
        return LLMResponse(model=model, text=text, raw=d, elapsed_s=elapsed)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def call(self, model: str, messages: list[dict], **kw) -> LLMResponse:
        """Single LLM call with retry. Returns LLMResponse (with .error if failed)."""
        self._check_budget()
        provider = _model_to_provider(model)
        if provider not in self.keys:
            return LLMResponse(model=model, text="",
                               error=f"No API keys for provider {provider!r}")

        _RATE_LIMIT_SIGNALS = ("ratelimit", "rate_limit", "429", "resource_exhausted",
                               "resourceexhausted", "toomanyrequests", "quota")
        _SERVER_ERROR_SIGNALS = ("connection timed out", "internal server error",
                                 "servererror: 500", " 502", " 503", " 504",
                                 "bad gateway", "service unavailable")

        def _is_rate_limit(err: str) -> bool:
            low = err.lower()
            return any(s in low for s in _RATE_LIMIT_SIGNALS)

        def _is_server_error(err: str) -> bool:
            low = err.lower()
            return any(s in low for s in _SERVER_ERROR_SIGNALS)

        last_err: str | None = None
        for attempt in range(self.max_retries):
            with self._acquire_key(provider) as key:
                try:
                    fn = getattr(self, f"_call_{provider}")
                    resp = fn(model, messages, key, **kw)
                    resp.n_attempts = attempt + 1
                    return resp
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    logger.warning(f"[{provider}/{model}] attempt {attempt+1} failed: {last_err}")
                    if _is_rate_limit(last_err):
                        delay = 10.0 + random.uniform(0, 5)
                        logger.info(f"  rate limit — waiting {delay:.1f}s before retry")
                    elif _is_server_error(last_err):
                        # Proxy-side 5xx (e.g. upstream connection timeout) — moderate delay
                        delay = 5.0 + random.uniform(0, 3) * (attempt + 1)
                        logger.info(f"  server error — waiting {delay:.1f}s before retry")
                    else:
                        delay = (self.backoff_base ** attempt) * (0.5 + random.random())
                    time.sleep(delay)

        return LLMResponse(model=model, text="", error=last_err, n_attempts=self.max_retries)

    def batch_call(
        self,
        model: str,
        prompts: list[list[dict]],
        max_workers: int = 100,
        progress: bool = True,
        **kw,
    ) -> list[LLMResponse]:
        """Concurrent batch of calls, preserves prompt order."""
        results: list[LLMResponse | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self.call, model, p, **kw): i
                for i, p in enumerate(prompts)
            }
            iterable: Iterable = as_completed(futures)
            if progress:
                try:
                    from tqdm import tqdm
                    iterable = tqdm(iterable, total=len(futures), desc=f"{model}")
                except ImportError:
                    pass
            for fut in iterable:
                i = futures[fut]
                results[i] = fut.result()
        return [r for r in results if r is not None]

    def __repr__(self) -> str:
        provider_summary = ", ".join(f"{p}({len(ks)})" for p, ks in self.keys.items())
        return f"APIKeyPool[{provider_summary}, cost=${self._cost_total_usd:.2f}]"


# Singleton convenience
_default_pool: APIKeyPool | None = None


def get_default_pool() -> APIKeyPool:
    """Return process-wide default pool, creating it on first call."""
    global _default_pool
    if _default_pool is None:
        _default_pool = APIKeyPool()
    return _default_pool
