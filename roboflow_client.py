"""
Roboflow workflow client (Serverless endpoint).

Works with URLs like:
https://serverless.roboflow.com/infer/workflows/<workflow-slug>/<block-name>

It sends a JSON payload:

    {
        "api_key": "...",
        "inputs": {
            "image": { "type": "base64", "value": "<b64>" }
        }
    }

and returns (annotated_image_bytes, counts_dict).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import aiohttp

# --------------------------------------------------------------------
# Set a default so the file works even if no env var is present.
# Replace with *your* workflow slug if you prefer.
DEFAULT_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "deer-appv1/detect-count-and-visualize-2"
)
# --------------------------------------------------------------------


class RoboflowClient:
    """Async client for a Roboflow *workflow*."""

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
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")
        if not self.workflow_url:
            raise ValueError("Workflow URL is not set.")

        # Log once so we know the correct URL is in use
        logging.info("Roboflow workflow URL: %s", self.workflow_url)

    # ------------------------------------------------------------------
    async def process_image(
        self,
        session: aiohttp.ClientSession,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> Tuple[bytes, Dict[str, int]]:
        """Send one image → get (annotated_bytes, counts)."""
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

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(data: Dict) -> Tuple[bytes, Dict[str, int]]:
        """
        Extract counts + annotated image from the workflow response.

        DEBUG: prints the first payload (truncate to 2 000 chars) to logs so
        we can see the exact schema and adjust if needed.
        """
        if not hasattr(RoboflowClient._parse_response, "_seen"):
            logging.info(
                "RAW ROBOFLOW RESPONSE:\n%s",
                json.dumps(data, indent=2)[:2000],
            )
            RoboflowClient._parse_response._seen = True

        # --------- count classes -------------------------------------------
        counts: Dict[str, int] = {"buck": 0, "deer": 0, "doe": 0}

        # Case 1: explicit "counts" object
        if isinstance(data.get("counts"), dict):
            for cls in counts:
                if isinstance(data["counts"].get(cls), int):
                    counts[cls] = int(data["counts"][cls])

        # Case 2: iterate over "predictions" list
        if not any(counts.values()):
            preds: List[Dict] = data.get("predictions") or data.get("results", [])
            for pred in preds:
                cls = (pred.get("class") or pred.get("label") or "").lower()
                if cls in counts:
                    counts[cls] += 1

        # --------- annotated image -----------------------------------------
        annotated_b64: Optional[str] = (
            data.get("image")
            or data.get("annotated_image")
            or data.get("media")
            or data.get("visualization")
        )
        annotated_bytes: bytes = b""
        if isinstance(annotated_b64, str) and annotated_b64:
            if "," in annotated_b64:  # strip data URI prefix if present
                annotated_b64 = annotated_b64.split(",", 1)[1]
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
            except Exception:
                annotated_bytes = b""

        return annotated_bytes, counts
