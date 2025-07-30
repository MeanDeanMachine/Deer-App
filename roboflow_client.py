"""
Asynchronous client for interacting with a **Roboflow *workflow***
deployed on the Serverless Inference endpoint.

Unlike the older `detect.roboflow.com/<project>/<version>` URLs (which
expect multipart form-data), *workflow* endpoints live at

    https://serverless.roboflow.com/infer/workflows/<workflow-slug>/<block-name>

and expect a **JSON** payload that includes your API key plus the image
encoded as base-64.

Environment variables
---------------------
ROBOFLOW_API_KEY
    Your Roboflow API key.

ROBOFLOW_WORKFLOW_URL   (preferred)
    The *full* workflow URL, e.g.
    https://serverless.roboflow.com/infer/workflows/deer-appv1/detect-count-and-visualize-2

If ROBOFLOW_WORKFLOW_URL is **unset**, the constructor falls back to a
hard-coded default.  Edit `DEFAULT_WORKFLOW_URL` to match your workflow.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional, Tuple

import aiohttp

# --------------------------------------------------------------------
# Adjust this to *your* workflow URL if you don’t want to rely on an
# environment variable:
DEFAULT_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "deer-appv1/detect-count-and-visualize-2"
)
# --------------------------------------------------------------------


class RoboflowClient:
    """Client for invoking a Roboflow *workflow* asynchronously."""

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

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------
    async def process_image(
        self,
        session: aiohttp.ClientSession,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> Tuple[bytes, Dict[str, int]]:
        """
        Send one image to the Roboflow workflow and return:

        1. The annotated image (bytes) – if the workflow returns it.
        2. A dict with counts for 'buck', 'deer', 'doe'.

        Parameters
        ----------
        session : aiohttp.ClientSession
            Shared session for HTTP requests.
        image_bytes : bytes
            Raw JPEG bytes.
        filename : str, optional
            Ignored by the workflow but useful for debugging.

        Raises
        ------
        aiohttp.ClientResponseError
            Propagated if the request fails (e.g. 401, 413, 5xx).
        """
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

    # -----------------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------------
import logging, json

@staticmethod
def _parse_response(data: Dict) -> Tuple[bytes, Dict[str, int]]:
    # ➜ DEBUG: print first payload once per session
    if not hasattr(_parse_response, "_seen"):
        logging.info("RAW ROBOFLOW RESPONSE:\n%s", json.dumps(data, indent=2)[:2000])
        _parse_response._seen = True
        """
        Extract counts and annotated image bytes from the workflow response.

        This works if your blocks output either:
        • `predictions` (list of dicts with 'class')
        • `counts` (dict with numeric counts)
        • `image` / `annotated_image` / `media` (base64 image string)

        Adjust as needed for your custom schema.
        """
        # --- count classes ---------------------------------------------------
        counts: Dict[str, int] = {"buck": 0, "deer": 0, "doe": 0}

        # Case 1: counts provided directly
        if isinstance(data.get("counts"), dict):
            direct = data["counts"]
            for cls in counts:
                if isinstance(direct.get(cls), int):
                    counts[cls] = int(direct[cls])

        # Case 2: tally predictions list
        if not any(counts.values()):
            preds: List[Dict] = data.get("predictions") or data.get("results", [])
            for pred in preds:
                cls = (pred.get("class") or pred.get("label") or "").lower()
                if cls in counts:
                    counts[cls] += 1

        # --- annotated image -------------------------------------------------
        annotated_b64: Optional[str] = (
            data.get("image") or data.get("annotated_image") or data.get("media")
        )
        annotated_bytes: bytes = b""
        if isinstance(annotated_b64, str) and annotated_b64:
            if "," in annotated_b64:  # strip data URI header
                annotated_b64 = annotated_b64.split(",", 1)[1]
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
            except Exception:
                annotated_bytes = b""

        return annotated_bytes, counts
