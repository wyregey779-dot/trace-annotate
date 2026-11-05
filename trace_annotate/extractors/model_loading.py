"""Offline-first model loading policy.

A vision adapter must never reach the network just because a weight path is
missing or misspelled. Offline mode is the default; an operator can opt into
downloads explicitly with TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD=1, which is
recorded in the adapter's load metadata so the provenance of the weights stays
visible.
"""
from __future__ import annotations

import os
from typing import Dict

ALLOW_DOWNLOAD_ENV = "TRACE_ANNOTATE_ALLOW_MODEL_DOWNLOAD"


def downloads_allowed() -> bool:
    """Downloads only when the operator explicitly opts in."""
    return os.environ.get(ALLOW_DOWNLOAD_ENV, "").strip() == "1"


def offline_mode() -> bool:
    return not downloads_allowed()


def enforce_offline_environment() -> Dict[str, str]:
    """Pin the hub clients to local files before any model import.

    huggingface_hub reads these variables when the request is made, so setting
    them here prevents the multi-minute connect-retry storm a missing weight used
    to trigger. Returns the effective values for provenance.
    """
    values = {}
    if offline_mode():
        for key, value in (("HF_HUB_OFFLINE", "1"),
                           ("TRANSFORMERS_OFFLINE", "1"),
                           ("HF_HUB_DISABLE_TELEMETRY", "1")):
            os.environ[key] = value
            values[key] = value
    else:
        values["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "0")
    return values


def load_policy() -> Dict[str, object]:
    """Describe the policy for model-run provenance records."""
    return {
        "offline": offline_mode(),
        "downloads_allowed": downloads_allowed(),
        "opt_in_env": ALLOW_DOWNLOAD_ENV,
    }


# huggingface_hub snapshots HF_HUB_OFFLINE into a module constant at import time,
# so the variables must be in place before anything imports it. Enforcing here
# means every adapter that imports this module is offline unless the operator
# opted in; a later os.environ write would be too late for the hub client.
ENFORCED_ENVIRONMENT = enforce_offline_environment()
