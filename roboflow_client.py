"""
Roboflow workflow client for DeerLens (Streamlit)

✓ Works with **Serverless Workflow** URLs  
✓ Falls back to legacy `detect.roboflow.com/<project>/<version>`  
✓ No batch limit – one image per HTTP call (Streamlit already loops)  
✓ Recursively extracts:
   • per-class `predictions` lists
   • `counts` dicts
   • Roboflow’s new `output_image` → {type:"base64", value:"…"} field
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import aiohttp

##############################################################################
DEFAULT_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "deer-appv1/detect-count-and-visualize-2"
)
DEFAULT_PROJECT_VER = ""  # For legacy mode, e.g. "deer-detect/1"
##############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

LEGACY_TEMPLATE = "https://detect.roboflow.com/{project_ver}?api_key={api_key}"
CLASSES = ("buck", "deer", "doe")


class RoboflowClient:
    """Async helper for Roboflow inference."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = "QlPxiNTzU6QV88y7xOj7",
        workflow_url: Optional[str] = None,
        legacy_project_ver: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY must be set.")

        self.workflow_url = (
            workflow_url
            or os.getenv("ROBOFLOW_WORKFLOW_URL")
            or DEFAULT_WORKFLOW_URL
        ).rstrip("/")

        self.legacy_project_ver = (
            legacy_project_ver
            or os.getenv("ROBOFLOW_PROJECT_VER")
            or DEFAULT_PROJECT_VER
        )

        self._use_workflow = "serverless.roboflow.com" in self.workflow_url
        logger.info(
            "Roboflow mode = %s | url = %s",
            "workflow" if self._use_workflow else "legacy",
            self.workflow_url
            if self._use_workflow
            else LEGACY_TEMPLATE.format(
                project_ver=self.legacy_project_ver or "<unset>", api_key="***"
            ),
        )

    # ------------------------------------------------------------------ #
    async def process_image(
        self,
        session: aiohttp.ClientSession,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> Tuple[bytes, Dict[str, int]]:
        if self._use_workflow:
            return await self._call_workflow(session, image_bytes)
        return await self._call_legacy(session, image_bytes, filename)

    # === Serverless workflow ========================================== #
    async def _call_workflow(
        self, session: aiohttp.ClientSession, img: bytes
    ) -> Tuple[bytes, Dict[str, int]]:
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {"type": "base64", "value": base64.b64encode(img).decode()}
            },
        }
        async with session.post(
            self.workflow_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
        ) as r:
            r.raise_for_status()
            data = await r.json()
            return self._parse_response(data)

    # === Legacy hosted model ========================================== #
    async def _call_legacy(
        self, session: aiohttp.ClientSession, img: bytes, filename: str
    ) -> Tuple[bytes, Dict[str, int]]:
        if not self.legacy_project_ver:
            raise ValueError("ROBOFLOW_PROJECT_VER must be set for legacy mode.")
        url = LEGACY_TEMPLATE.format(
            project_ver=self.legacy_project_ver, api_key=self.api_key
        )
        form = aiohttp.FormData()
        form.add_field("file", img, filename=filename, content_type="image/jpeg")
        async with session.post(url, data=form) as r:
            r.raise_for_status()
            data = await r.json()
            return self._parse_response(data)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(data: Union[Dict, List]) -> Tuple[bytes, Dict[str, int]]:
        """
        Recursively gather:
        • class counts  (from `predictions` list or `counts` dict)
        • annotated image (from `output_image.value` or any *_image key)
        """

        # Dump first payload for debugging
        if not getattr(RoboflowClient, "_dumped", False):
            logger.info("RAW ROBOFLOW RESPONSE (trunc 2 k):\n%s", json.dumps(data)[:2000])
            RoboflowClient._dumped = True

        counts = {cls: 0 for cls in CLASSES}
        image_b64: Optional[str] = None

        def walk(node: Union[Dict, List]) -> None:
            nonlocal image_b64
            if isinstance(node, dict):
                # ---- class counts -----------------------------------------
                if "counts" in node and isinstance(node["counts"], dict):
                    for cls in CLASSES:
                        if isinstance(node["counts"].get(cls), (int, float)):
                            counts[cls] += int(node["counts"][cls])

                preds = node.get("predictions") or node.get("detections")
                if isinstance(preds, list):
                    for p in preds:
                        cls = (p.get("class") or p.get("label") or "").lower()
                        if cls in counts:
                            counts[cls] += 1

                # ---- annotated image --------------------------------------
                if image_b64 is None:
                    # New workflow field: output_image {type:"base64", value:"..."}
                    if (
                        "output_image" in node
                        and isinstance(node["output_image"], dict)
                        and node["output_image"].get("type") == "base64"
                        and isinstance(node["output_image"].get("value"), str)
                    ):
                        image_b64 = node["output_image"]["value"]

                    if image_b64 is None:
                        for key in ("image", "annotated_image", "media", "visualization"):
                            if isinstance(node.get(key), str):
                                image_b64 = node[key]
                                break

                # ---- recurse further --------------------------------------
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        walk(v)

            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)

        # decode / return
        img_bytes = b""
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(image_b64)
            except Exception:
                pass

        return img_bytes, counts
