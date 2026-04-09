"""
minigrok.base
Logging, metadata, and core flags.
"""
import json, logging, os

APP_NAME    = "MiniGrok"
APP_VERSION = "13.0"

# ── Logging ─────────────────────────────────────────────────────
log = logging.getLogger("MiniGrok")
log.setLevel(logging.INFO)
log.propagate = False   # Prevent duplicate output from root logger

class _JSONFormatter(logging.Formatter):
    """Emit log records as one-line JSON — machine-queryable."""
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts":    self.formatTime(record, datefmt="%H:%M:%S"),
            "level": record.levelname,
            "msg":   record.getMessage(),
        }
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)

# Single handler: JSON for INFO+, with a readable fallback line for WARNING+
# This prevents the triple-print issue (JSON × 2 + plain text) seen in Colab
if not log.handlers:   # Guard against re-run in same notebook session
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JSONFormatter())
    log.addHandler(_handler)

log.info("Imports OK")
