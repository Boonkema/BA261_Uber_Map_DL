"""Optional BA261 activity: explore New York City Uber pickups with Streamlit.

Source data and original concept: Streamlit's Uber NYC tutorial
https://docs.streamlit.io/get-started/tutorials/create-an-app
https://github.com/streamlit/demo-uber-nyc-pickups

Workshop guidance: Chris L. Santo, Scottsdale Community College,
NITIC Summer Working Connections, Advanced Data Analytics with Python, APIs and AI,
Columbus, Ohio, June 1-4, 2026.

Animated heat map and related classroom exploration:
Professor Brian Daily, BA261, with AI-assisted development.

This demonstration uses public, historical sample data from September 2014.
"""

import time

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="NYC Uber Pickup Explorer", page_icon="🚕", layout="wide")

DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_data(show_spinner="Loading the example pickup data (about 1 million rows)...")
def load_data(number_of_rows=1_100_000):
    """Load the public dataset, no API key or local file needed.

    The September 2014 file has roughly one million rows, so asking for
    1,100,000 loads the entire month.
    """
    data = pd.read_csv(DATA_URL, nrows=number_of_rows)
    data.columns = [str(name).lower() for name in data.columns]
    data["date/time"] = pd.to_datetime(data["date/time"], errors="coerce")
    data = data.dropna(subset=["date/time", "lat", "lon"]).copy()
    data["date"] = data["date/time"].dt.date
    data["hour"] = data["date/time"].dt.hour
    data["weekday"] = data["date/time"].dt.day_name()
    return data


@st.cache_data(show_spinner=False)
def weekday_hour_averages(data):
    """Average pickups for each weekday/hour combination.

    We count pickups for each calendar date first, then average across dates,
    so weekdays that appear five times in September are not overstated.
    """
    per_day = data.groupby(["date", "weekday", "hour"]).size().reset_index(name="n")
    return (
        per_day.groupby(["weekday", "hour"])["n"]
        .mean()
        .reset_index(name="Avg pickups")
    )


def make_map(points, visualization="heat"):
    """Make either a 2D density heat map or a 3D hexagon map."""
    points = points[["lat", "lon"]]
    if visualization == "3d":
        layer = pdk.Layer(
            "HexagonLayer",
            data=points,
            get_position="[lon, lat]",
            radius=210,
            elevation_scale=15,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        )
    else:
        layer = pdk.Layer(
            "HeatmapLayer",
            data=points,
            get_position="[lon, lat]",
            get_weight=1,
            radius_pixels=55,
        )
    return pdk.Deck(
        map_style=None,  # Match the Streamlit theme. No API key required.
        initial_view_state=pdk.ViewState(
            latitude=40.75,
            longitude=-73.98,
            zoom=10.5,
            pitch=42 if visualization == "3d" else 0,
        ),
        layers=[layer],
    )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🚕 Uber pickups in New York City")
st.write(
    "Explore *historical* Uber pickup locations from September 2014. "
    "Choose a date and hour, then watch how concentrations of pickups "
    "change during the day."
)
st.caption("Demonstration project for BA261 — no programming background required to explore it.")

try:
    data = load_data()
except Exception as error:
    st.error("The example dataset could not be downloaded at this time.")
    st.info("Check the internet connection or try again later. The data source is linked below.")
    st.stop()

# ---------------------------------------------------------------------------
# Date and hour selection
# ---------------------------------------------------------------------------
available_dates = sorted(data["date"].unique())
selected_date = st.selectbox(
    "Date to explore", available_dates, format_func=lambda d: d.strftime("%B %d, %Y")
)
selected_hour = st.slider("Hour of day (24-hour clock)", 0, 23, 17)
daily_data = data[data["date"] == selected_date]
hour_data = daily_data[daily_data["hour"] == selected_hour]

m1, m2, m3 = st.columns(3)
m1.metric("Pickups in selected date", f"{len(daily_data):,}")
m2.metric("Pickups in selected hour", f"{len(hour_data):,}")
m3.metric("Time shown", f"{selected_hour:02d}:00–{(selected_hour + 1):02d}:00")

# ---------------------------------------------------------------------------
# Pickups by hour (bar chart)
# ---------------------------------------------------------------------------
st.subheader("Pickups by hour")
counts = np.bincount(daily_data["hour"].to_numpy(), minlength=24)
hourly_counts = pd.DataFrame({"Hour": range(24), "Pickups": counts})
st.bar_chart(hourly_counts.set_index("Hour"))

# ---------------------------------------------------------------------------
# Map for the selected hour
# ---------------------------------------------------------------------------
st.subheader("Map for your selected hour")
map_kind = st.radio("Map style", ["2D heat map", "3D hexagons"], horizontal=True)
st.pydeck_chart(
    make_map(hour_data, "3d" if map_kind == "3D hexagons" else "heat"), height=440
)
st.caption("Bright clusters or tall columns indicate higher concentrations of pickup locations.")

# ---------------------------------------------------------------------------
# NEW: Hour-by-weekday heat map
# ---------------------------------------------------------------------------
st.subheader("Average pickups by weekday and hour")
st.write(
    "Each square shows the average number of pickups for that weekday and hour "
    "across all of September 2014. Compare weekday rush hours with weekend nights."
)
heat_data = weekday_hour_averages(data)
weekday_chart = (
    alt.Chart(heat_data)
    .mark_rect()
    .encode(
        x=alt.X("hour:O", title="Hour of day (24-hour clock)"),
        y=alt.Y("weekday:O", sort=DAY_ORDER, title=None),
        color=alt.Color(
            "Avg pickups:Q", scale=alt.Scale(scheme="viridis"), title="Avg pickups"
        ),
        tooltip=[
            alt.Tooltip("weekday:N", title="Day"),
            alt.Tooltip("hour:O", title="Hour"),
            alt.Tooltip("Avg pickups:Q", title="Avg pickups", format=",.0f"),
        ],
    )
    .properties(height=300)
)
st.altair_chart(weekday_chart, use_container_width=True)
st.caption("Lighter squares mean more pickups. Hover over a square to see its exact average.")

# ---------------------------------------------------------------------------
# NEW: Compare two times of day side by side
# ---------------------------------------------------------------------------
st.subheader("Compare two times of day")
st.write(
    f"Pick two hours to compare on **{selected_date:%B %d, %Y}** "
    "(the date chosen above). Try morning rush hour next to late night."
)
map_style_compare = st.radio(
    "Map style for comparison",
    ["2D heat map", "3D hexagons"],
    horizontal=True,
    key="compare_style",
)
compare_kind = "3d" if map_style_compare == "3D hexagons" else "heat"

col_a, col_b = st.columns(2)
hour_a = col_a.slider("Hour A", 0, 23, 8, key="hour_a")
hour_b = col_b.slider("Hour B", 0, 23, 22, key="hour_b")

data_a = daily_data[daily_data["hour"] == hour_a]
data_b = daily_data[daily_data["hour"] == hour_b]

col_a.markdown(f"**{hour_a:02d}:00** — {len(data_a):,} pickups")
col_b.markdown(f"**{hour_b:02d}:00** — {len(data_b):,} pickups")
col_a.pydeck_chart(make_map(data_a, compare_kind), height=380, key="compare_map_a")
col_b.pydeck_chart(make_map(data_b, compare_kind), height=380, key="compare_map_b")

# ---------------------------------------------------------------------------
# 24-hour animation
# ---------------------------------------------------------------------------
st.subheader("▶ 24-hour animated heat map")
st.write(
    "Press Play to watch the selected date progress from midnight (00:00) "
    "through 11 p.m. (23:00). The animation runs once and then stops."
)
seconds_per_hour = st.slider("Animation speed (seconds per hour)", 0.2, 1.0, 0.5, 0.1)
if st.button("▶ Play one full day"):
    status_area = st.empty()
    movie_area = st.empty()
    for hour in range(24):
        current = daily_data[daily_data["hour"] == hour]
        status_area.markdown(
            f"**{selected_date:%B %d, %Y} — {hour:02d}:00** — {len(current):,} pickups"
        )
        movie_area.pydeck_chart(make_map(current), height=440, key="daily_animation_map")
        time.sleep(seconds_per_hour)
    st.success("The 24-hour animation is complete. Press Play to watch it again.")

# ---------------------------------------------------------------------------
# Raw data and sources
# ---------------------------------------------------------------------------
if st.checkbox("Show the first 100 data rows"):
    st.dataframe(daily_data.head(100))

st.markdown("---")
st.caption(
    "Source: Streamlit's public Uber pickup tutorial and September 2014 dataset. "
    "This app loads up to 1,100,000 records, which covers the entire September 2014 file. "
    "Patterns in a historical sample do not establish causes."
)
st.markdown(
    "[Streamlit tutorial](https://docs.streamlit.io/get-started/tutorials/create-an-app) | "
    "[Original demo repository and dataset](https://github.com/streamlit/demo-uber-nyc-pickups)"
)
