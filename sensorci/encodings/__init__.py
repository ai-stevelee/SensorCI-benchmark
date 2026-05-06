"""Three text encodings for feeding signals into LLMs (per A0 attack)."""

from sensorci.encodings.raw_stats import encode_raw_stats
from sensorci.encodings.sax import encode_sax
from sensorci.encodings.compact import encode_compact_numeric

ENCODERS = {
    "raw_stats": encode_raw_stats,
    "sax": encode_sax,
    "compact_numeric": encode_compact_numeric,
}

__all__ = ["encode_raw_stats", "encode_sax", "encode_compact_numeric", "ENCODERS"]
