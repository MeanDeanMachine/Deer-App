"""
Asynchronous client for interacting with a Roboflow inference workflow.

The Roboflow API can return predictions for uploaded images along with an
annotated image.  This module defines a ``RoboflowClient`` class that
encapsulates the logic for submitting a single JPEG image to a workflow
endpoint and parsing the response.  Calls are made using ``aiohttp`` to
allow many images to be processed concurrently.

Environment variables
=====================

Two environment variables control how the client connects to the API:

``ROBOFLOW_API_KEY``
    Your Roboflow API key.  Do **not** hard‑code this value; instead set
    it in your shell environment or via a `.env` file.  The API key is
    appended as a query parameter when invoking the workflow.

``ROBOFLOW_WORKFLOW_ID``
    The identifier of your workflow.  This corresponds to the workflow
    slug configured in your Roboflow workspace.  The endpoint URL is
    built from this value.  Refer to Roboflow's documentation for the
    exact format of workflow endpoints.

Response format assumptions
===========================

The code assumes that the response JSON includes:

* A ``predictions`` field containing a list of bounding boxes.  Each
  prediction should have a ``class`` key whose value is one of
  ``"buck"``, ``"deer"`` or ``"doe"``:contentReference[oaicite:3]{index=3}.  The client counts the number
  of predictions per class.

* An ``image`` field containing a base64 encoded image string prefixed
  with a data URI (e.g. ``"data:image/jpeg;base64,...."``) representing
  the annotated image.  The client decodes this image into raw bytes for
  display.  If your workflow instead returns a binary blob when
  ``format=image`` is requested:contentReference[oaicite:4]{index=4}, you may need to adjust the
  parsing logic accordingly.

If your workflow returns different fields (for example ``counts``
containing class counts directly, or a different key for the annotated
image), you will need to adjust the ``_parse_response`` method accordingly.
"""

from __future__ import annotations

import base64
import os
from typing import Dict, Tuple, Optional, List

import aiohttp


class RoboflowClient:
    """Client for invoking Roboflow inference workflows asynchronously."""

    def __init__(self, api_key: Optional[str] = None, workflow_id: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.workflow_id = workflow_id or os.getenv("ROBOFLOW_WORKFLOW_ID")
        if not self.api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable is not set."
            )
        if not self.workflow_id:
            raise ValueError(
                "ROBOFLOW_WORKFLOW_ID environment variable is not set."
            )
        # Base URL for workflow invocation.  Adjust this template to match
        # Roboflow's API specification for workflows.  See the README for
        # details.
        # TODO: Replace the placeholder with the correct base URL for your workflow
        self.base_url = f"https://detect.roboflow.com/{self.workflow_id}"

    async def process_image(self, session: aiohttp.ClientSession, image_bytes: bytes, filename: str) -> Tuple[bytes, Dict[str, int]]:
        """Submit an image to the workflow and return the annotated image and class counts.

        Parameters
        ----------
        session:
            An existing :class:`aiohttp.ClientSession` used to make the request.
        image_bytes:
            The raw JPEG bytes to send to Roboflow.
        filename:
            The filename of the image.  This is sent as part of the multipart
            form payload and may help with diagnostics on the server side.

        Returns
        -------
        Tuple[bytes, Dict[str, int]]
            A tuple containing the annotated image bytes and a dictionary of
            counts keyed by class name.
        """
        url = f"{self.base_url}?api_key={self.api_key}"
        # Construct a multipart/form-data payload
        form = aiohttp.FormData()
        form.add_field(
            name="file",
            value=image_bytes,
            filename=filename,
            content_type="image/jpeg",
        )
        async with session.post(url, data=form) as response:
            response.raise_for_status()
            data = await response.json()
            annotated_image, counts = self._parse_response(data)
            return annotated_image, counts

    @staticmethod
    def _parse_response(data: Dict) -> Tuple[bytes, Dict[str, int]]:
        """Extract the annotated image and counts from the API response.

        This method encapsulates all assumptions about the structure of the
        Roboflow response.  If the API returns a different schema, adjust
        this method to suit.

        Parameters
        ----------
        data:
            The parsed JSON response from Roboflow.

        Returns
        -------
        Tuple[bytes, Dict[str, int]]
            Raw bytes of the annotated image and a mapping of class names
            (``"buck"``, ``"deer"``, ``"doe"``) to their respective counts.
        """
        # Attempt to extract counts from a "predictions" field
        counts = {"buck": 0, "deer": 0, "doe": 0}
        preds: List[Dict] = data.get("predictions") or data.get("results", [])
        if isinstance(preds, list):
            for pred in preds:
                clazz = pred.get("class") or pred.get("label")
                if clazz in counts:
                    counts[clazz] += 1
        # Some workflows may return a counts object directly
        if not any(counts.values()) and isinstance(data.get("counts"), dict):
            counts_data = data["counts"]
            for clazz in counts:
                if clazz in counts_data and isinstance(counts_data[clazz], int):
                    counts[clazz] = counts_data[clazz]
        # Extract annotated image as bytes.  Expect a base64 data URI.
        annotated_b64: Optional[str] = data.get("image") or data.get("annotated_image") or data.get("media")
        annotated_bytes: bytes = b""
        if isinstance(annotated_b64, str) and annotated_b64:
            # Strip any data URI prefix (e.g. 'data:image/jpeg;base64,')
            if "," in annotated_b64:
                annotated_b64 = annotated_b64.split(",", 1)[1]
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
            except Exception:
                annotated_bytes = b""
        return annotated_bytes, counts
