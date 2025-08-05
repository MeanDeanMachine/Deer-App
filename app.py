"""
DeerLens – Streamlit front-end
==============================
* Upload up to 900 JPEGs
* Runs Roboflow workflow asynchronously
* Shows KPI cards, stacked bar, time-of-day heat-map
* Clickable 800-px thumbnails
* **NEW:** Inline editor lets you adjust mis-classified counts, instantly
           refreshing all visuals and the downloadable CSV.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import io                       # gives io.StringIO
import aiohttp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sequential
import streamlit as st

from io import BytesIO
from PIL import Image          # thumbnail creation

from exif_utils import extract_datetime_original
from roboflow_client import RoboflowClient

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DIRECTIONS = ["", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]  # "" = no entry

# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ImageResult:
    file_name: str
    date_time: Optional[str]
    buck_count: int
    deer_count: int
    doe_count: int
    annotated_image: bytes
    direction: Optional[str] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Async inference helpers
# ──────────────────────────────────────────────────────────────────────
async def _process_single(
    session: aiohttp.ClientSession,
    rf_client: RoboflowClient,
    file_name: str,
    image_bytes: bytes,
) -> ImageResult:
    """Send one image to Roboflow and return counts + annotated JPEG."""
    try:
        date_time = extract_datetime_original(image_bytes)
    except Exception:
        date_time = None

    try:
        annotated, counts = await rf_client.process_image(
            session, image_bytes, file_name
        )
           # ── NEW: shrink annotated JPEG to ≤600 px width to save RAM ──
        try:
            img = Image.open(BytesIO(annotated))
            img.thumbnail((600, 600), Image.LANCZOS)      # long edge ≤600
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            annotated = buf.getvalue()
        except Exception:
            # If Pillow fails, fall back to original bytes
            pass

        return ImageResult(
            file_name=file_name,
            date_time=date_time,
            buck_count=counts.get("buck", 0),
            deer_count=counts.get("deer", 0),
            doe_count=counts.get("doe", 0),
            annotated_image=annotated,
        )
    except Exception as exc:
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
    files: List[Tuple[str, bytes]], rf_client: RoboflowClient
) -> List[ImageResult]:
    """Concurrently process many images."""
    results, processed, total = [], 0, len(files)
    bar = st.progress(0)
    sem = asyncio.Semaphore(3)

    async with aiohttp.ClientSession() as session:

        async def task(name: str, data: bytes) -> ImageResult:
            async with sem:
                try:       
                    return await _process_single(session, rf_client, name, data)
                finally:
                   # free the raw upload immediately
                    del data

        tasks = [task(n, b) for n, b in files]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            results.append(res)
            processed += 1
            bar.progress(processed / total)
    bar.empty()
    return results


# ──────────────────────────────────────────────────────────────────────
# Data wrangling
# ──────────────────────────────────────────────────────────────────────
def to_dataframe(results: List[ImageResult]) -> pd.DataFrame:
    """Results ➜ tidy DataFrame."""
    return pd.DataFrame(
        [
            dict(
                file_name=r.file_name,
                date_time=r.date_time,
                buck_count=r.buck_count,
                deer_count=r.deer_count,
                doe_count=r.doe_count,
                direction=r.direction
            )
            for r in results
        ]
    )


def compute_summary(df: pd.DataFrame) -> Dict[str, int]:
    return dict(
        total_images=len(df),
        total_buck=int(df["buck_count"].sum()),
        total_deer=int(df["deer_count"].sum()),
        total_doe=int(df["doe_count"].sum()),
    )


def bucket_time(ts: pd.Timestamp | None) -> str | None:
    if pd.isna(ts):
        return None
    m = ts.hour * 60 + ts.minute
    if 360 <= m <= 480:
        return "Dawn"
    if 481 <= m <= 660:
        return "Morning"
    if 661 <= m <= 960:
        return "Midday"
    if 961 <= m <= 1110:
        return "Afternoon"
    if 1111 <= m <= 1259:
        return "Evening"
    return "Night"


def categorise(results: List[ImageResult], df_counts: pd.DataFrame) -> Dict[str, List[ImageResult]]:
    """Bucket each ImageResult according to edited counts."""
    latest = df_counts.set_index("file_name")
    cat: Dict[str, List[ImageResult]] = {"Buck": [], "Deer": [], "Doe": [], "No Tag": []}
    for res in results:
        if res.file_name not in latest.index:
            cat["No Tag"].append(res)
            continue
        row = latest.loc[res.file_name]
        counts = {"Buck": row.buck_count, "Deer": row.deer_count, "Doe": row.doe_count}
        mx = max(counts.values())
        if mx == 0:
            cat["No Tag"].append(res)
        elif counts["Buck"] == mx:
            cat["Buck"].append(res)
        elif counts["Deer"] == mx:
            cat["Deer"].append(res)
        else:
            cat["Doe"].append(res)
    return cat


# ──────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DeerLens", layout="wide")
st.title("🦌 DeerLens – Deer Detection and Analysis")
st.markdown(
    """
Upload between **1 and 900** JPEG images from your trail cam.
DeerLens detects deer, draws bounding boxes, and gives you instant stats.
"""
)

uploader = st.file_uploader(
    "Select JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True
)
error_box = st.empty()

# run pipeline ---------------------------------------------------------
if uploader:
    if not 1 <= len(uploader) <= 900:
        st.warning("Select 1-900 images.")
        st.stop()

    if st.button("Process Images"):
        with st.spinner("Calling Roboflow…"):
            # Pre-load all JPEG bytes (list stays in RAM only until we clear it)
            files_data = [(f.name, f.read()) for f in uploader]

            try:
                client = RoboflowClient()
            except Exception as e:
                st.error(str(e))
                st.stop()

            # async inference
            results = asyncio.run(process_images_async(files_data, client))

            files_data.clear()          # <<< free raw upload bytes

        # cache results for UI / editing
        st.session_state["image_results"] = results
        st.session_state["orig_df"]     = to_dataframe(results)
        st.session_state["edited_df"]   = st.session_state["orig_df"].copy()

# show editor / analytics if results exist ----------------------------
if "edited_df" in st.session_state:
    df = st.session_state["edited_df"]
    results = st.session_state["image_results"]

    # editable grid ----------------------------------------------------
    st.subheader("Review / correct counts")
    edited = st.data_editor(
        df,
        disabled=["file_name", "date_time"],
        key="editor",
        num_rows="fixed",
        use_container_width=True,
    )
    if st.button("Apply overrides", key="apply"):
        st.session_state["edited_df"] = edited
        df = edited  # refresh local reference

    # KPI cards --------------------------------------------------------
    kpi_container = st.empty()

    # stacked bar ------------------------------------------------------
    if df["date_time"].notna().any():
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        bar_df = df.copy()
        bar_df["date"] = dt.dt.date
        agg = (
            bar_df.groupby("date", as_index=False)
            .sum(numeric_only=True)
            .sort_values("date")
        )
        melt = agg.melt("date", ["buck_count", "deer_count", "doe_count"],
                        var_name="Class", value_name="Count")
        melt["Class"] = melt["Class"].str.replace("_count", "").str.title()
        fig = px.bar(
            melt,
            x="date",
            y="Count",
            color="Class",
            color_discrete_map={
                "Buck": "#228B22",
                "Doe": "#FFC0CB",
                "Deer": "#D2B48C",
            },
            title="Activity by Date (stacked)",
            labels={"date": "Date"},
        )
        fig.update_layout(barmode="stack", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # heat-map ---------------------------------------------------------
    if df["date_time"].notna().any():
        ts = pd.to_datetime(df["date_time"], errors="coerce")
        heat_df = df.copy()
        heat_df["date"] = ts.dt.date
        heat_df["bucket"] = ts.apply(bucket_time)

        agg = (
            heat_df.groupby(["bucket", "date"], as_index=False)
            .agg(buck=("buck_count", "sum"),
                 deer=("deer_count", "sum"),
                 doe=("doe_count", "sum"))
        )
        agg["activity"] = agg["buck"] + agg["deer"] + agg["doe"]

        buckets = ["Dawn", "Morning", "Midday", "Afternoon", "Evening", "Night"]
        dates = sorted(agg["date"].unique())

        z = [[int(agg.query("bucket==@b and date==@d")["activity"].sum())
              for d in dates] for b in buckets]

        sx, sy, ss, sc = [], [], [], []
        for b in buckets:
            for d in dates:
                bk = int(agg.query("bucket==@b and date==@d")["buck"].sum())
                if bk:
                    sx.append(d); sy.append(b); ss.append(bk * 10 + 8); sc.append(bk)

        heat = go.Heatmap(
            z=z, x=dates, y=buckets,
            colorscale=sequential.Blues,
            colorbar=dict(title="Total Activity"),
            hovertemplate="Date %{x}<br>%{y}<br>Activity %{z}<extra></extra>",
        )
        dots = go.Scatter(
            x=sx, y=sy, mode="markers",
            marker=dict(size=ss, color="#228B22", opacity=.85,
                        line=dict(width=1, color="white")),
            customdata=sc,  # buck count
            hovertemplate="Date %{x}<br>%{y}<br>Bucks %{customdata}<extra></extra>",
            showlegend=False,
        )
        fig_hm = go.Figure([heat, dots])
        fig_hm.update_yaxes(autorange="reversed")
        fig_hm.update_layout(
            title=("Heat-map of Total Activity<br>"
                   "<span style='font-size:0.8em'>(bubble size = buck count)</span>"),
            xaxis_tickangle=-45, xaxis_type="category",
            yaxis_title="Time of Day", xaxis_title="Date",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # download ---------------------------------------------------------
    # 1) summary → DataFrame
    summary_df = pd.DataFrame(
        [
            {"metric": "total_images", "value": smry["total_images"]},
            {"metric": "total_buck",   "value": smry["total_buck"]},
            {"metric": "total_deer",   "value": smry["total_deer"]},
            {"metric": "total_doe",    "value": smry["total_doe"]},
        ]
    )

    # 2) build one CSV: summary, blank line, detailed rows
    buf = io.StringIO()
    summary_df.to_csv(buf, index=False)
    buf.write("\n")                 # visual separator
    st.session_state["edited_df"].to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    st.download_button(
        "Download revised CSV (+ summary)",
        csv_bytes,
        "deerlens_results.csv",
        "text/csv",
    )
    # remember the last category pane the user opened
    if "open_cat" not in st.session_state:
        st.session_state.open_cat = None
    # gallery ----------------------------------------------------------
    st.header("Annotated Images")

    THUMB_W = 800
    cats = categorise(results, st.session_state.edited_df)

    for cat_name in ["Buck", "Deer", "Doe", "No Tag"]:
        imgs = cats.get(cat_name, [])

        # sort by date_time (earliest first)
        imgs = sorted(
             imgs,
             key=lambda r: pd.to_datetime(r.date_time, errors="coerce")
        )
               
        # remember if this pane should be open on rerun
        expanded_now = st.session_state.open_cat == cat_name
        with st.expander(f"{cat_name} ({len(imgs)})", expanded=expanded_now):
            # keep this pane open after user clicks inside it
            if st.session_state.open_cat != cat_name:
                st.session_state.open_cat = cat_name

            if not imgs:
                st.write("No images.")
                continue

            for res in imgs:
                if not res.annotated_image:
                    st.write(f"{res.file_name} (no annotated image)")
                    continue

                # ------------- thumbnail ---------------------------------
                b64 = base64.b64encode(res.annotated_image).decode()
                uri = f"data:image/jpeg;base64,{b64}"

                # current counts from edited_df
                row = st.session_state.edited_df.loc[
                    st.session_state.edited_df["file_name"] == res.file_name
                ].iloc[0]

                col_img, col_edit = st.columns([3, 2])

                with col_img:
                    # thumbnail + caption
                    st.image(
                        res.annotated_image,
                        width=THUMB_W,
                        caption=res.file_name,
                        output_format="JPEG",
                    )

                    # click-to-zoom (opens image inside about:blank)
                    st.components.v1.html(
                        f"""
                        <a href="#" onclick="
                             const w = window.open('about:blank');
                             w.document.write(`<img src='{uri}' style='width:100%;'>`);
                             return false;
                        ">🔍 Open full size</a>
                        """,
                        height=24,
                        scrolling=False,
                    )

                with col_edit:
                    st.markdown("**Manual overrides**")

                    # ------------- inline form ---------------------------
                    dir_options = DIRECTIONS[1:]                             # exclude blank sentinel
                    current_dir = row.direction if pd.notna(row.direction) else ""
                    with st.form(key=f"form_{res.file_name}", clear_on_submit=False):
                        buck_val = st.number_input(
                            "Buck", min_value=0, value=int(row.buck_count)
                        )
                        deer_val = st.number_input(
                            "Deer", min_value=0, value=int(row.deer_count)
                        )
                        doe_val = st.number_input(
                            "Doe", min_value=0, value=int(row.doe_count)
                        )
                        dir_val = st.selectbox(
                            "Direction (optional)",
                            options=["—"] + dir_options,
                            index=(dir_options.index(current_dir) + 1) if current_dir else 0,
                            key=f"dir_{res.file_name}",
                        )
                        submitted = st.form_submit_button(
                            "Save", use_container_width=True
                        )

                    # apply edits when user clicks Save
                    if submitted:
                        dir_clean = None if dir_val == "—" else dir_val
                        st.session_state.edited_df.loc[
                            st.session_state.edited_df["file_name"] == res.file_name,
                            ["buck_count", "deer_count", "doe_count", "direction"],
                        ] = [buck_val, deer_val, doe_val, dir_clean]
                        st.rerun()
# ──────────────────────────────────────────────────────────────────
#  🔄  Refresh KPI cards & build up-to-date CSV  🔄
#      (runs after gallery so it always has the newest edits)
# ──────────────────────────────────────────────────────────────────
df_final   = st.session_state["edited_df"]
smry_final = compute_summary(df_final)

# --- repopulate KPI container ------------------------------------
with kpi_container:
    st.header("Summary Metrics")
    cols = st.columns(4)
    cols[0].metric("Images", smry_final["total_images"])
    cols[1].metric("Buck",   smry_final["total_buck"])
    cols[2].metric("Deer",   smry_final["total_deer"])
    cols[3].metric("Doe",    smry_final["total_doe"])

# --- build CSV (summary + detailed rows) -------------------------
summary_df = pd.DataFrame(
    [
        {"metric": "total_images", "value": smry_final["total_images"]},
        {"metric": "total_buck",   "value": smry_final["total_buck"]},
        {"metric": "total_deer",   "value": smry_final["total_deer"]},
        {"metric": "total_doe",    "value": smry_final["total_doe"]},
    ]
)

buf = io.StringIO()
summary_df.to_csv(buf, index=False)
buf.write("\n")                      # visual separator
df_final.to_csv(buf, index=False)
csv_bytes = buf.getvalue().encode("utf-8")

st.download_button(
    "Download revised CSV (+ summary)",
    csv_bytes,
    "deerlens_results.csv",
    "text/csv",
)
