"""
Async client for both Roboflow Serverless *workflows* **and** the older
`detect.roboflow.com` upload endpoint.

★ Usage -----------------------------------------------------------------
from roboflow_client import RoboflowClient
client = RoboflowClient()                         # picks up env vars
annot_jpg, counts = await client.process_image(session, jpeg_bytes)

Env vars ---------------------------------------------------------------
ROBOFLOW_API_KEY          (required)
ROBOFLOW_WORKFLOW_URL     e.g.
  https://serverless.roboflow.com/infer/workflows/deer-appv1/detect-count-and-visualize-2
  – if omitted we will try legacy mode instead.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import aiohttp

# ----------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------

LEGACY_TEMPLATE = "https://detect.roboflow.com/{project_ver}?api_key={api_key}"
DEFAULT_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "deer-appv1/detect-count-and-visualize-2"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class RoboflowClient:
    """Asynchronous helper for Roboflow inference."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        workflow_url: Optional[str] = None,
        legacy_project_ver: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY must be set in environment")

        self.workflow_url = (
            workflow_url
            or os.getenv("ROBOFLOW_WORKFLOW_URL")
            or DEFAULT_WORKFLOW_URL
        ).rstrip("/")

        # For the *old* endpoint you still need <project>/<version>
        self.legacy_project_ver = (
            legacy_project_ver or os.getenv("ROBOFLOW_PROJECT_VER")
        )

        # Decide which mode we are in
        self._use_workflow = "serverless.roboflow.com" in self.workflow_url

        logger.info(
            "RoboflowClient initialised – mode=%s, url=%s",
            "workflow" if self._use_workflow else "legacy",
            self.workflow_url if self._use_workflow else LEGACY_TEMPLATE.format(
                project_ver=self.legacy_project_ver or "<unset>…", api_key="***"
            ),
        )

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------
    async def process_image(
        self,
        session: aiohttp.ClientSession,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> Tuple[bytes, Dict[str, int]]:
        """
        Execute one inference request.

        Returns
        -------
        (annotated_bytes, counts)
        """
        if self._use_workflow:
            return await self._call_workflow(session, image_bytes)
        # fallback – legacy detect.roboflow.com
        return await self._call_legacy(session, image_bytes, filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _call_workflow(
        self, session: aiohttp.ClientSession, img: bytes
    ) -> Tuple[bytes, Dict[str, int]]:
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": base64.b64encode(img).decode("ascii"),
                }
            },
        }
        headers = {"Content-Type": "application/json"}
        async with session.post(
            self.workflow_url, headers=headers, data=json.dumps(payload)
        ) as r:
            r.raise_for_status()
            data = await r.json()
            return self._parse_response(data)

    async def _call_legacy(
        self,
        session: aiohttp.ClientSession,
        img: bytes,
        filename: str,
    ) -> Tuple[bytes, Dict[str, int]]:
        if not self.legacy_project_ver:
            raise ValueError(
                "ROBOFLOW_PROJECT_VER (project/version) must be set for legacy mode"
            )
        url = LEGACY_TEMPLATE.format(
            project_ver=self.legacy_project_ver, api_key=self.api_key
        )
        form = aiohttp.FormData()
        form.add_field(
            "file",
            img,
            filename=filename,
            content_type="image/jpeg",
        )
        async with session.post(url, data=form) as r:
            r.raise_for_status()
            data = await r.json()
            return self._parse_response(data)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(data: Dict) -> Tuple[bytes, Dict[str, int]]:
        # Dump **once** per session for debugging
        if not getattr(RoboflowClient, "_dumped", False):
            logger.info("RAW ROBOFLOW RESPONSE (truncated):\n%s",
                        json.dumps(data)[:1500])
            RoboflowClient._dumped = True

        counts: Dict[str, int] = {"buck": 0, "deer": 0, "doe": 0}

        # 1️⃣ direct counts
        if isinstance(data.get("counts"), dict):
            for k in counts:
                counts[k] = int(data["counts"].get(k, 0))

        # 2️⃣ flattened predictions
        if not any(counts.values()):
            preds: List[Dict] = (
                data.get("predictions")
                or data.get("results")
                or data.get("outputs", {}).get("detections", [])
            )
            for p in preds or []:
                cls = (p.get("class") or p.get("label") or "").lower()
                if cls in counts:
                    counts[cls] += 1

        # 3️⃣ annotated image (optional)
        annotated_b64: Optional[str] = (
            data.get("image")
            or data.get("annotated_image")
            or data.get("media")
        )
        annot_bytes = b""
        if annotated_b64:
            if "," in annotated_b64:  # strip data URI prefix
                annotated_b64 = annotated_b64.split(",", 1)[1]
            try:
                annot_bytes = base64.b64decode(annotated_b64)
            except Exception:
                pass

        return annot_bytes, counts
