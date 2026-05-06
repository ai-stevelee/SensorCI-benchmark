"""Verify environment setup before running pilots.

Checks:
  - Python version
  - All sensorci imports
  - .env loaded with at least one API provider
  - Optional: ping each provider with a 1-token request
"""

import sys
import os
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check(name: str, ok: bool, detail: str = "") -> bool:
    sym = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{sym}] {name}{(': ' + detail) if detail else ''}")
    return ok


def main(ping: bool = False) -> int:
    print("=== SensorCI environment check ===\n")
    all_ok = True

    # 1. Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 11)
    all_ok &= check(
        "Python version",
        py_ok,
        f"{py_version.major}.{py_version.minor}.{py_version.micro} (need >=3.11)",
    )

    # 2. Imports
    print("\nImports:")
    try:
        import numpy, scipy, pandas, yaml, tqdm, dotenv
        all_ok &= check("numpy/scipy/pandas/yaml/tqdm/dotenv", True)
    except ImportError as e:
        all_ok &= check("numpy/scipy/pandas/yaml/tqdm/dotenv", False, str(e))

    try:
        import openai, anthropic
        all_ok &= check("openai + anthropic SDKs", True)
    except ImportError as e:
        all_ok &= check("openai + anthropic SDKs", False, str(e))

    try:
        from google import genai
        all_ok &= check("google-genai", True)
    except ImportError as e:
        check("google-genai", False, str(e) + " (optional but recommended)")

    try:
        import sensorci
        from sensorci.core import Persona, Segment, TSDataset
        from sensorci.api_pool import APIKeyPool
        from sensorci.datasets import load_dataset
        from sensorci.defenses import get_defense
        from sensorci.attacks import get_attack
        all_ok &= check("sensorci package", True, f"v{sensorci.__version__}")
    except ImportError as e:
        all_ok &= check("sensorci package", False, str(e))
        return 1  # Hard fail

    # 3. .env loaded
    print("\nAPI keys:")
    from dotenv import load_dotenv
    # scripts/00_check_env.py is at: 002_SensorCI/code/scripts/
    # .env is at:                    002_SensorCI/  (3 levels up)
    _candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",  # 002_SensorCI/
        Path(__file__).parent.parent / ".env",         # code/
    ]
    env_path = next((p for p in _candidates if p.exists()), _candidates[0])
    load_dotenv(env_path, override=False)

    hub_base = os.environ.get("HUB_BASE_URL", os.environ.get("AZURE_OPENAI_BASE_URL", ""))
    check("  API hub HUB_BASE_URL", bool(hub_base), hub_base[:60] if hub_base else "not set")

    # Count numbered Azure keys (AZURE_OPENAI_API_KEY_1..N)
    hub_keys = [v for i in range(1, 201) if (v := os.environ.get(f"AZURE_OPENAI_API_KEY_{i}", "").strip())]
    # Also check comma-separated vars
    for env_var in ("OPENAI_API_KEYS", "ANTHROPIC_API_KEYS", "GOOGLE_API_KEYS", "HUB_API_KEYS"):
        v = os.environ.get(env_var, "").strip()
        if v:
            hub_keys += [k for k in v.split(",") if k.strip()]
    hub_keys = list(dict.fromkeys(hub_keys))  # deduplicate
    n_hub = len(hub_keys)
    all_ok &= check("  API hub API keys", n_hub > 0, f"{n_hub} key(s) (Azure numbered + explicit vars)")

    if n_hub == 0:
        print(f"\n{YELLOW}Warning: no API keys found. Pilot scripts will fail.{RESET}")
        print(f"  Edit {env_path} — ensure AZURE_OPENAI_API_KEY_1..N or HUB_API_KEYS is set")
        all_ok = False

    def _first(env_var: str) -> str:
        """Return the first model from a possibly comma-separated env var."""
        val = os.environ.get(env_var, "").strip()
        return val.split(",")[0].strip() if val else ""

    azure_model = _first("AZURE_OPENAI_MODEL")
    claude_model = _first("CLAUDE_MODEL")
    gemini_model = _first("GEMINI_MODEL")
    check("  AZURE_OPENAI_MODEL", bool(azure_model), azure_model or "not set")
    check("  CLAUDE_MODEL (first)", bool(claude_model), claude_model or "not set")
    check("  GEMINI_MODEL (first)", bool(gemini_model), gemini_model or "not set")
    n_providers_with_keys = 1 if n_hub > 0 else 0

    # 4. Optional ping — one model per provider
    if ping and n_providers_with_keys > 0:
        print("\nPinging API hub providers (1 token each)...")
        from sensorci.api_pool import APIKeyPool
        pool = APIKeyPool(providers=("openai", "anthropic", "google"))
        ping_models = [m for m in [azure_model, claude_model, gemini_model] if m]
        for model in ping_models:
            try:
                r = pool.call(model, [{"role": "user", "content": "say OK"}], max_tokens=5)
                check(f"  ping {model}", r.error is None,
                      r.text[:40] if not r.error else r.error)
            except Exception as e:
                check(f"  ping {model}", False, str(e))

    # 5. Data + results dirs
    print("\nDirectories:")
    data_dir = Path(os.environ.get("SENSORCI_DATA_DIR", "./data"))
    results_dir = Path(os.environ.get("SENSORCI_RESULTS_DIR", "./results"))
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    check(f"  data dir {data_dir}", data_dir.exists())
    check(f"  results dir {results_dir}", results_dir.exists())

    print()
    if all_ok:
        print(f"{GREEN}All checks passed. Ready to run pilots.{RESET}")
        return 0
    else:
        print(f"{RED}Some checks failed. Fix before running pilots.{RESET}")
        return 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true",
                    help="Send a 1-token request to each provider to verify keys")
    args = ap.parse_args()
    sys.exit(main(ping=args.ping))
