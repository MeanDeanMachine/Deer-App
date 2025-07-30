# DeerLens

DeerLens is a single‑user Streamlit web application for detecting and
counting deer in batches of JPEG images.  It uploads images directly
from your SD card, invokes a **Roboflow inference workflow** to
annotate them, extracts metadata from the original photos and presents
aggregate statistics and visualisations.  A download button allows you
to save the raw results as a CSV file, and an image gallery neatly
organises the annotated outputs.

## Quick start

1. **Clone the repository** (or copy the code files into a directory).

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
