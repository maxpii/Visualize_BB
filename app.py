import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


DATA_FILE = "iNaturalist_Branch Brook Park Survey Data 2026_ADHC_FYS.csv"


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    df = pd.read_csv(path, encoding="utf-8-sig")

    if df.empty:
        return df

    for col in [
        "latitude",
        "longitude",
        "observed_on",
        "quality_grade",
        "common_name",
        "scientific_name",
        "iconic_taxon_name",
        "description",
        "image_url",
        "url",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["common_name"] = df["common_name"].replace(r"^\s*$", np.nan, regex=True)
    df["scientific_name"] = df["scientific_name"].replace(r"^\s*$", np.nan, regex=True)
    df["iconic_taxon_name"] = df["iconic_taxon_name"].replace(r"^\s*$", np.nan, regex=True)
    df["quality_grade"] = df["quality_grade"].fillna("unknown")
    df["description"] = df["description"].fillna("No notes available")
    df["image_url"] = df["image_url"].fillna("")
    df["url"] = df["url"].fillna("")

    df["common_name"] = df["common_name"].fillna(df["scientific_name"]).fillna("Unidentified")
    df["scientific_name"] = df["scientific_name"].fillna(df["common_name"]).fillna("Unidentified")
    df["iconic_taxon_name"] = df["iconic_taxon_name"].fillna("Unidentified")

    df["quality_label"] = df["quality_grade"].str.lower().str.replace("_", " ").str.title()
    df["quality_label"] = df["quality_label"].replace("Needs Id", "Needs ID")
    df["quality_label"] = df["quality_label"].replace("Research", "Research Grade")

    return df


@st.cache_data
def filter_data(df: pd.DataFrame, selected_categories, selected_quality, start_date, end_date):
    mask = pd.Series(True, index=df.index)

    if selected_categories and "All" not in selected_categories:
        mask &= df["iconic_taxon_name"].isin(selected_categories)

    if selected_quality:
        allowed = {"Research Grade", "Needs ID"}
        selected_quality_set = set(selected_quality)
        valid_filters = selected_quality_set & allowed
        if valid_filters:
            grade_map = {
                "Research Grade": ["research", "research grade"],
                "Needs ID": ["needs_id", "needs id"],
            }
            allow_values = []
            for label in valid_filters:
                allow_values.extend(grade_map.get(label, []))
            mask &= df["quality_grade"].fillna("").str.lower().isin(allow_values)

    if start_date is not None:
        mask &= df["observed_on"].dt.date >= start_date
    if end_date is not None:
        mask &= df["observed_on"].dt.date <= end_date

    return df.loc[mask].copy()


def safe_title(name):
    if pd.isna(name) or str(name).strip() == "":
        return "Unidentified"
    return str(name).strip()


def hex_to_rgb(value: str):
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) for i in range(0, 6, 2))


def build_category_colors(categories):
    palette = [
        "#166534",
        "#C2410C",
    ]
    return {category: palette[index % len(palette)] for index, category in enumerate(categories)}


def render_map(filtered_df: pd.DataFrame):
    valid = filtered_df.dropna(subset=["latitude", "longitude"]).copy()

    if valid.empty:
        st.info("No location data matches the current filters.")
        return

    category_colors = build_category_colors(sorted(valid["iconic_taxon_name"].unique().tolist()))

    m = folium.Map(
        location=[
            valid["latitude"].mean(),
            valid["longitude"].mean(),
        ],
        zoom_start=15,
        tiles="CartoDB dark_matter" if st.get_option("theme.base") == "dark" else "CartoDB positron",
    )

    for _, row in valid.iterrows():
        category = safe_title(row["iconic_taxon_name"])
        common_name = safe_title(row["common_name"])
        scientific_name = safe_title(row["scientific_name"])
        obs_date = row["observed_on"].strftime("%Y-%m-%d") if pd.notna(row["observed_on"]) else "Unknown date"
        color = category_colors.get(category, "#7ec8e3")

        popup_html = f"""
        <div style='font-family: sans-serif; min-width: 220px;'>
            <strong>{common_name}</strong><br>
            <em>{scientific_name}</em><br>
            <span>Category: {category}</span><br>
            <span>Date: {obs_date}</span>
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{common_name} · {scientific_name} · {obs_date}",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    st_folium(m, width=700, height=500)


def render_gallery(filtered_df: pd.DataFrame):
    if filtered_df.empty:
        st.info("No observations match the filters.")
        return

    cols = st.columns(3)
    for index, row in filtered_df.reset_index(drop=True).head(18).iterrows():
        col = cols[index % 3]
        with col:
            image_url = row.get("image_url", "")
            if isinstance(image_url, str) and image_url.strip():
                st.image(image_url, use_column_width=True)
            else:
                st.markdown(
                    "<div style='height: 180px; display:flex; align-items:center; justify-content:center; border:1px solid #111827; border-radius: 12px; background:#ffffff; color:#111827; font-weight:600;'>No image</div>",
                    unsafe_allow_html=True,
                )

            common_name = safe_title(row.get("common_name", ""))
            scientific_name = safe_title(row.get("scientific_name", ""))
            description = str(row.get("description", "No notes available")).strip() or "No notes available"
            obs_date = row.get("observed_on")
            date_str = obs_date.strftime("%Y-%m-%d") if pd.notna(obs_date) else "Unknown date"

            st.markdown(f"### {common_name}")
            st.caption(f"{scientific_name} · {date_str}")
            st.write(description[:220] + ("..." if len(description) > 220 else ""))
            if row.get("url"):
                st.link_button("View on iNaturalist", row["url"], use_container_width=True)
            else:
                st.button("No observation link", disabled=True, use_container_width=True)
            st.markdown("---")


def main():
    st.set_page_config(page_title="Branch Brook Park Survey Explorer", layout="wide")

    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] { background: #ffffff; color: #111827; }
        [data-testid="stHeader"] { background: #ffffff; }
        [data-testid="stSidebar"] { background: #f3f4f6; }
        .stApp,
        .stApp p,
        .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"] { color: #111827; }
        .stApp h1,
        .stApp h2,
        .stApp h3 { color: #111827; }
        .block-container { padding-top: 1.5rem; }
        .metric-card {
            background: #ffffff;
            border: 1px solid #111827;
            border-top: 4px solid #166534;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            color: #111827;
        }
        .metric-label { color: #111827; font-size: 0.9rem; font-weight: 600; }
        .metric-value { color: #111827; font-size: 2rem; font-weight: 700; }
        .stApp [data-testid="stBaseButton-primary"],
        .stApp [data-testid="stLinkButton"] a {
            background: #c2410c;
            border: 1px solid #9a3412;
            color: #ffffff;
        }
        .stApp [data-testid="stBaseButton-primary"]:hover,
        .stApp [data-testid="stLinkButton"] a:hover {
            background: #9a3412;
            border-color: #7c2d12;
            color: #ffffff;
        }
        .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div,
        .stApp [data-baseweb="datepicker"] {
            background: #ffffff;
            border-color: #374151;
            color: #111827;
        }
        .stApp [data-baseweb="tab-list"] button { color: #111827; }
        .stApp [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #9a3412;
            border-bottom-color: #c2410c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        st.error(f"Dataset not found: {data_path.name}. Please place the CSV file in the same folder as this app.")
        st.stop()

    df = load_data(str(data_path))

    if df.empty:
        st.warning("The CSV file is empty or could not be read.")
        st.stop()

    st.title("Branch Brook Park Observation Explorer")
    st.caption("Interactive biodiversity and iNaturalist survey dashboard")

    with st.sidebar:
        st.header("Filters")

        categories = ["All"] + sorted(df["iconic_taxon_name"].dropna().unique().tolist())
        selected_categories = st.multiselect("Category", categories, default=["All"])

        quality_options = ["Research Grade", "Needs ID"]
        selected_quality = st.multiselect("Quality Grade", quality_options, default=quality_options)

        observed_dates = df["observed_on"].dropna()
        if not observed_dates.empty:
            min_date = observed_dates.min().date()
            max_date = observed_dates.max().date()
            start_date, end_date = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            start_date = end_date = None
            st.info("No valid observation dates available.")

    filtered_df = filter_data(df, selected_categories, selected_quality, start_date, end_date)

    total_obs = len(filtered_df)
    unique_species = filtered_df["scientific_name"].nunique()
    top_category = filtered_df["iconic_taxon_name"].value_counts().idxmax() if not filtered_df.empty else "N/A"
    research_count = filtered_df["quality_grade"].fillna("").str.lower().eq("research").sum()
    pct_research = (research_count / total_obs * 100) if total_obs else 0

    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        f"<div class='metric-card'><div class='metric-label'>Total Observations</div><div class='metric-value'>{total_obs}</div></div>",
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        f"<div class='metric-card'><div class='metric-label'>Unique Species</div><div class='metric-value'>{unique_species}</div></div>",
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        f"<div class='metric-card'><div class='metric-label'>Top Category</div><div class='metric-value'>{top_category}</div></div>",
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        f"<div class='metric-card'><div class='metric-label'>% Research Grade</div><div class='metric-value'>{pct_research:.1f}%</div></div>",
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning("No observations match the selected filters. Try broadening your filter choices.")
        st.stop()

    tab_map, tab_stats = st.tabs(["Map", "Analytics"])

    with tab_map:
        render_map(filtered_df)

    with tab_stats:
        category_counts = (
            filtered_df["iconic_taxon_name"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Category", "iconic_taxon_name": "Count"})
        )
    
        category_counts.columns = ["Category", "Count"]
        category_counts = category_counts.sort_values("Count", ascending=False)

        species_counts = (
            filtered_df["scientific_name"]
            .fillna("Unidentified")
            .value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"index": "Species", "scientific_name": "Count"})
        )
        species_counts.columns = ["Species", "Count"]
        species_counts = species_counts.sort_values("Count", ascending=False)

        charts_col1, charts_col2 = st.columns(2)
        with charts_col1:
            st.subheader("Observations by Category")
            st.bar_chart(category_counts.set_index("Category"), use_container_width=True)
        with charts_col2:
            st.subheader("Top 10 Species")
            st.bar_chart(species_counts.set_index("Species"), use_container_width=True)

    st.subheader("Observation Gallery")
    render_gallery(filtered_df)

    with st.expander("Raw dataset preview"):
        st.dataframe(filtered_df.head(200), use_container_width=True)


if __name__ == "__main__":
    main()
