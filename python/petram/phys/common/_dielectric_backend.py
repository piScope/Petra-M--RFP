"""Process-wide dielectric backend selection, read once at first import."""
import os

_value = os.environ.get("PETRAM_RFP_USE_NUMBA_DIELECTRIC", "0").strip().lower()
if _value in ("1", "true", "yes", "on"):
    USE_NUMBA = True
elif _value in ("0", "false", "no", "off", ""):
    USE_NUMBA = False
else:
    raise ValueError("PETRAM_RFP_USE_NUMBA_DIELECTRIC must be 0/1 or false/true")
BACKEND = "numba" if USE_NUMBA else "ext"
