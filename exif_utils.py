"""
Exif utilities for the DeerLens application.

This module exposes a single helper to extract the original date/time
metadata from JPEG images.  By default it uses the `exifread` Python
package which can parse EXIF headers directly from an in‑memory byte
stream.  If `exifread` is unavailable or fails to produce a result, a
fallback to the system‑installed ExifTool can be used.  ExifTool must
be installed and available on the system `PATH` for the fallback to
work.  In practice, the primary route (exifread) suffices for most
cases and avoids spawning a subprocess for every file.

The returned date string is normalised to the format
``YYYY-MM-DD HH:MM:SS``.  If no valid timestamp can be extracted the
function returns ``None``.
"""

from __future__ import annotations

import io
import os
import subprocess
from datetime import datetime
from typing import Optional

try:
    import exifread  # type: ignore
except ImportError:
    exifread = None


def _parse_datetime(value: str) -> Optional[str]:
    """Normalise an EXIF DateTime string.

    EXIF tags typically encode timestamps as ``YYYY:MM:DD HH:MM:SS``.  This
    helper converts the first two colons to dashes.  If the value cannot
    be parsed it returns ``None``.
    """
    try:
        # Replace only the first two colons (between year/month and month/day)
        parts = value.strip().split(" ")
        if len(parts) != 2:
            return None
        date_part, time_part = parts
        # There should be exactly two colons in the date part
        date_segments = date_part.split(":")
        if len(date_segments) != 3:
            return None
        normalised_date = "-".join(date_segments)
        # Validate the timestamp by parsing with datetime
        datetime.strptime(f"{normalised_date} {time_part}", "%Y-%m-%d %H:%M:%S")
        return f"{normalised_date} {time_part}"
    except Exception:
        return None


def extract_datetime_original(data: bytes) -> Optional[str]:
    """Extract the EXIF DateTimeOriginal from raw JPEG bytes.

    Parameters
    ----------
    data:
        The raw contents of the JPEG file as a ``bytes`` object.

    Returns
    -------
    Optional[str]
        A string in the format ``YYYY-MM-DD HH:MM:SS`` if the tag is
        present and can be parsed, otherwise ``None``.
    """
    # Attempt to use exifread if available
    if exifread is not None:
        try:
            with io.BytesIO(data) as buffer:
                tags = exifread.process_file(buffer, stop_tag="EXIF DateTimeOriginal", details=False)
                # EXIF DateTimeOriginal may be stored under different key names depending on the camera
                for key in ("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"):
                    if key in tags:
                        raw_value = str(tags[key])
                        normalised = _parse_datetime(raw_value)
                        if normalised:
                            return normalised
        except Exception:
            # silently fall through to the fallback
            pass

    # Fallback: call exiftool via subprocess.  This requires exiftool to be installed
    # and accessible on the system PATH.  The command extracts the DateTimeOriginal
    # tag and formats it as YYYY-MM-DD HH:MM:SS.
    try:
        with subprocess.Popen(
            [
                "exiftool",
                "-DateTimeOriginal",
                "-d",
                "%Y-%m-%d %H:%M:%S",
                "-s",
                "-s",
                "-s",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            stdout_data, _ = proc.communicate(data, timeout=10)
            if proc.returncode == 0 and stdout_data:
                value = stdout_data.decode("utf-8", errors="ignore").strip()
                # exiftool returns a single value if found, otherwise empty string
                if value:
                    # exiftool output is already in desired format
                    return value
    except Exception:
        pass

    return None


__all__ = ["extract_datetime_original"]
