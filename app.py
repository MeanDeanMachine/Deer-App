"""
Main Streamlit application for the DeerLens project.

This script wires together the user interface and business logic for
uploading images, performing inference via Roboflow, extracting EXIF
metadata, aggregating results and visualising them.  The app is designed
to operate on batches of up to 900 JPEG images uploaded through a file
uploader.  All processing is performed asynchronously to maximise
throughput and provide responsive progress feedback.

To run the app locally:

1. Create a virtual environment (optional but recommended).
2. Install dependencies with ``pip install -r requirements.txt``.
3. Set the environment variables ``ROBOFLOW_API_KEY`` and
   ``ROBOFLOW_WORKFLOW_ID`` in your shell.
4. Execute ``streamlit run app.py``.

Refer to the accompanying README for more detailed setup instructions.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import aiohttp

from roboflow_client import RoboflowClient
from exif_utils import extract_datetime_original


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    """Container for per‑image inference results."""
    file_name: str
    date_time: Optional[str]
    buck_count: int
    deer_count: int
    doe_count: int
    annotated_image: bytes
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _process_single(
    session: aiohttp.ClientSession,
    rf_client: RoboflowClient,
    file_name: str,
    image_bytes: bytes,
) -> ImageResult:
    """Process a single image through Roboflow and extract EXIF metadata.

    This helper is intended to be called within an asyncio event loop.

    Parameters
    ----------
    session:
        A shared :class:`aiohttp.ClientSession` for making HTTP requests.
    rf_client:
        A :class:`RoboflowClient` instance configured with API key and workflow ID.
    file_name:
        The original filename of the image.
    image_bytes:
        Raw JPEG bytes of the image.

    Returns
    -------
    ImageResult
        The result containing counts, timestamp and annotated image.
    """
    try:
        # Extract EXIF DateTimeOriginal
        date_time = extract_datetime_original(image_bytes)
    except Exception:
        # Log but do not fail the entire image processing
        date_time = None

    try:
        annotated_image, counts = await rf_client.process_image(session, image_bytes, file_name)
        buck_count = counts.get("buck", 0)
        deer_count = counts.get("deer", 0)
        doe_count = counts.get("doe", 0)
        return ImageResult(
            file_name=file_name,
            date_time=date_time,
            buck_count=buck_count,
            deer_count=deer_count,
            doe_count=doe_count,
            annotated_image=annotated_image,
        )
    except Exception as exc:
        # Capture the exception and return a result with zero counts
        return ImageResult(
            file_name=file_name,
            date_time=date_time,
            buck_count=0,
            deer_count=0,
            doe_count=0,
            annotated_image=b"",
            error=str(exc),
        )


async def process_images_async(
    files: List[Tuple[str, bytes]],
    rf_client: RoboflowClient,
) -> List[ImageResult]:
    """Process multiple images concurrently and return their results.

    Parameters
    ----------
    files:
        A list of tuples ``(file_name, image_bytes)``.
    rf_client:
        An instance of :class:`RoboflowClient` used to invoke the inference endpoint.

    Returns
    -------
    List[ImageResult]
        A list of results corresponding to the input files, in no particular order.
    """
    results: List[ImageResult] = []
    total = len(files)
    # Display a progress bar that updates as each image finishes processing
    progress_bar = st.progress(0)
    processed = 0

    semaphore = asyncio.Semaphore(5)  # Limit concurrency to avoid overwhelming the API
    async with aiohttp.ClientSession() as session:

        async def wrapped_process(file_name: str, image_bytes: bytes) -> ImageResult:
            async with semaphore:
                return await _process_single(session, rf_client, file_name, image_bytes)

        # Kick off all tasks
        tasks = [wrapped_process(fname, data) for fname, data in files]
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            processed += 1
            progress_bar.progress(processed / total)
    progress_bar.empty()
    return results


def aggregate_results(results: List[ImageResult]) -> pd.DataFrame:
    """Convert a list of ImageResult objects into a tidy DataFrame."""
    records = []
    for res in results:
        records.append(
            {
                "file_name": res.file_name,
                "date_time": res.date_time,
                "buck_count": res.buck_count,
                "deer_count": res.deer_count,
                "doe_count": res.doe_count,
                "any_tagged": bool(res.buck_count or res.deer_count or res.doe_count),
            }
        )
    df = pd.DataFrame(records)
    return df


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, any]:
    """Compute summary statistics and top dates from the DataFrame."""
    summary: Dict[str, any] = {}
    summary["total_images"] = len(df)
    summary["total_buck"] = int(df["buck_count"].sum())
    summary["total_deer"] = int(df["deer_count"].sum())
    summary["total_doe"] = int(df["doe_count"].sum())

    # Convert date_time strings to date objects for grouping.  Missing dates become NaT.
    if df["date_time"].notna().any():
        dt_series = pd.to_datetime(df["date_time"], errors="coerce")
        df_dates = df.copy()
        df_dates["date"] = dt_series.dt.date
        top_dates: Dict[str, pd.DataFrame] = {}
        for cls in ["buck", "deer", "doe"]:
            counts_per_date = (
                df_dates.groupby("date")[f"{cls}_count"].sum().nlargest(5)
            )
            top_dates[cls] = counts_per_date.reset_index().rename(columns={"index": "date", f"{cls}_count": "count"})
        summary["top_dates"] = top_dates
    else:
        summary["top_dates"] = {cls: pd.DataFrame(columns=["date", "count"]) for cls in ["buck", "deer", "doe"]}
    return summary


def categorise_results(results: List[ImageResult]) -> Dict[str, List[ImageResult]]:
    """Assign each image to one of four categories based on its highest count.

    The categories are ``Buck``, ``Deer``, ``Doe`` and ``No Tag``.
    Ties are resolved by the order Buck → Deer → Doe.  Images with zero
    counts in all categories are assigned to ``No Tag``.
    """
    categories: Dict[str, List[ImageResult]] = {"Buck": [], "Deer": [], "Doe": [], "No Tag": []}
    for res in results:
        # Determine which class has the maximum count
        counts = {"Buck": res.buck_count, "Deer": res.deer_count, "Doe": res.doe_count}
        max_count = max(counts.values())
        if max_count == 0:
            categories["No Tag"].append(res)
        else:
            # In case of ties, priority order is Buck > Deer > Doe
            if counts["Buck"] == max_count:
                categories["Buck"].append(res)
            elif counts["Deer"] == max_count:
                categories["Deer"].append(res)
            else:
                categories["Doe"].append(res)
    return categories


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DeerLens", layout="wide")
st.title("🦌 DeerLens – Deer Detection and Analysis")
st.markdown(
    """
    Upload between **1 and 900** JPEG images captured from your trail camera and
    let DeerLens automatically identify deer and compute summary statistics.
    All processing runs locally in this session and no images are stored
    permanently.
    """
)


# File uploader
uploaded_files = st.file_uploader(
    label="Select JPEG images",
    type=["jpg", "jpeg"],
    accept_multiple_files=True,
    help="Hold Ctrl/Cmd to select multiple files."
)

# Container for errors to display at the end
error_placeholder = st.empty()

if uploaded_files:
    num_files = len(uploaded_files)
    if num_files < 1:
        st.warning(
            f"Please select at least 1 image. You have selected {num_files} file(s)."
        )
    elif num_files > 900:
        st.warning(
            f"You have selected {num_files} images which exceeds the 900 image limit."
        )
    else:
        if st.button("Process Images", key="process_button"):
            with st.spinner("Processing images, please wait..."):
                # Preload file bytes to avoid reading the same file object in multiple threads
                files_data: List[Tuple[str, bytes]] = []
                for uploaded_file in uploaded_files:
                    try:
                        file_bytes = uploaded_file.read()
                        files_data.append((uploaded_file.name, file_bytes))
                    except Exception:
                        # Skip files that cannot be read
                        st.warning(f"Unable to read file: {uploaded_file.name}")

                # Instantiate the Roboflow client (will throw if env vars are missing)
                try:
                    rf_client = RoboflowClient()
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

                # Run the asynchronous processing loop
                results: List[ImageResult] = asyncio.run(
                    process_images_async(files_data, rf_client)
                )

                # Display errors (if any) at the top of the page
                failed = [r for r in results if r.error]
                if failed:
                    error_messages = "\n".join(f"{r.file_name}: {r.error}" for r in failed)
                    error_placeholder.warning(
                        f"The following files could not be processed:\n{error_messages}"
                    )

                # Aggregate into a DataFrame
                df = aggregate_results(results)

                # Compute summary statistics
                summary = compute_summary_stats(df)

                # Show KPI metrics
                st.header("Summary Metrics")
                cols = st.columns(4)
                cols[0].metric("Images Processed", summary["total_images"])
                cols[1].metric("Buck Count", summary["total_buck"])
                cols[2].metric("Deer Count", summary["total_deer"])
                cols[3].metric("Doe Count", summary["total_doe"])

                # Bar chart for total counts
                fig = px.bar(
                    x=["Buck", "Deer", "Doe"],
                    y=[summary["total_buck"], summary["total_deer"], summary["total_doe"]],
                    labels={"x": "Class", "y": "Count"},
                    title="Total Counts by Class",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Top 5 dates per class
                st.header("Top 5 Dates per Class")
                top_dates = summary.get("top_dates", {})
                date_cols = st.columns(3)
                for idx, cls in enumerate(["buck", "deer", "doe"]):
                    sub_df = top_dates.get(cls, pd.DataFrame())
                    with date_cols[idx]:
                        st.subheader(cls.capitalize())
                        if not sub_df.empty:
                            st.table(sub_df)
                        else:
                            st.write("No data available")

                # CSV download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="deerlens_results.csv",
                    mime="text/csv",
                )

                # Annotated image gallery
                st.header("Annotated Images")
                categories = categorise_results(results)
                for category_name in ["Buck", "Deer", "Doe", "No Tag"]:
                    images = categories.get(category_name, [])
                    with st.expander(f"{category_name} ({len(images)})", expanded=False):
                        for res in images:
                            if res.annotated_image:
                                # Display the annotated image with a uniform width
                                st.image(
                                    res.annotated_image,
                                    caption=f"{res.file_name} | {res.date_time or 'Unknown'}",
                                    width=300,
                                )
                            else:
                                # If there is no annotated image due to error, show placeholder
                                st.write(f"{res.file_name} (no image)")
