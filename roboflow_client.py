"""
Roboflow *workflow* client for Streamlit DeerLens
-------------------------------------------------

• Works with **Serverless** workflow URLs
  https://serverless.roboflow.com/infer/workflows/<workflow-slug>/<block-name>

• Sends JSON with the image as **base-64**.

• Automatically extracts class counts (`buck`, `deer`, `doe`) and an
  annotated image regardless of whether they live at top level *or*
  nested inside `results` → <block>.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
ROBOFLOW_API_KEY          – required.
ROBOFLOW_WORKFLOW_URL     – full workflow URL (preferred).
If the latter is absent we fall back to DEFAULT_WORKFLOW_URL below.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import aiohttp

##############################################################################
# Edit this constant if you don’t want to set ROBOFLOW_WORKFLOW_URL.
##############################################################################
DEFAULT_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "deer-appv1/detect-count-and-visualize-2"
)
##############################################################################


class RoboflowClient:
    """Async helper to call a Roboflow *workflow*."""

    CLASSES = ("buck", "deer", "doe")  # canonical order / names

    def __init__(
        self,
        api_key: Optional[str] = None,
        workflow_url: Optional[str] = None,
    ) -> None:
        self.api_key: str = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.workflow_url: str = (
            workflow_url
            or os.getenv("ROBOFLOW_WORKFLOW_URL")
            or DEFAULT_WORKFLOW_URL
        )
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set")
        if not self.workflow_url:
            raise ValueError("Workflow URL is not set")

        # Visible in Streamlit Cloud logs, so we know the right code is running
        print(f"RoboflowClient using URL: {self.workflow_url}")

    # --------------------------------------------------------------------- #
    async def process_image(
        self,
        session: aiohttp.ClientSession,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> Tuple[bytes, Dict[str, int]]:
        """Return (annotated_bytes, counts) for one image."""
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": base64.b64encode(image_bytes).decode("ascii"),
                }
            },
        }
        headers = {"Content-Type": "application/json"}

        async with session.post(
            self.workflow_url, headers=headers, data=json.dumps(payload)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return self._parse_response(data)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _parse_response(data: Dict) -> Tuple[bytes, Dict[str, int]]:
        """Robustly pull counts + annotated image from arbitrary schema."""
        # DEBUG – print only once per app start
        if not hasattr(RoboflowClient._parse_response, "_seen"):
            print("▼▼▼ RAW ROBOFLOW RESPONSE ▼▼▼")
            print(json.dumps(data, indent=2)[:4000])
            print("▲▲▲ END RAW RESPONSE ▲▲▲")
            RoboflowClient._parse_response._seen = True

        # ---- counts -------------------------------------------------------
        counts: Dict[str, int] = {k: 0 for k in RoboflowClient.CLASSES}

        def _accumulate(obj: Dict) -> None:
            """Add counts from a nested dict if present."""
            if not isinstance(obj, dict):
                return
            if "counts" in obj and isinstance(obj["counts"], dict):
                for cls in RoboflowClient.CLASSES:
                    if isinstance(obj["counts"].get(cls), (int, float)):
                        counts[cls] += int(obj["counts"][cls])
            if "predictions" in obj and isinstance(obj["predictions"], list):
                for pred in obj["predictions"]:
                    cls = (pred.get("class") or pred.get("label") or "").lower()
                    if cls in counts:
                        counts[cls] += 1

        # top-level first
        _accumulate(data)

        # look one level under 'results'
        if "results" in data and isinstance(data["results"], dict):
            for block in data["results"].values():
                _accumulate(block)

        # ---- annotated image ---------------------------------------------
        annotated_b64: Optional[str] = None

        def _maybe_set_image(obj: Dict) -> None:
            nonlocal annotated_b64
            if annotated_b64 is not None:
                return
            for key in ("image", "annotated_image", "media", "visualization"):
                if isinstance(obj.get(key), str):
                    annotated_b64 = obj[key]
                    return

        _maybe_set_image(data)
        if "results" in data and isinstance(data["results"], dict):
            for block in data["results"].values():
                _maybe_set_image(block)

        annotated_bytes: bytes = b""
        if isinstance(annotated_b64, str):
            if "," in annotated_b64:  # strip data URI scheme
                annotated_b64 = annotated_b64.split(",", 1)[1]
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
            except Exception:
                annotated_bytes = b""

        return annotated_bytes, counts
