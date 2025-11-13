import os
from typing import Optional
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text


# ---------------
# Cache helpers
# ---------------
@st.cache_resource(show_spinner=False)
def get_engine(uri: str):
    """Create or reuse a SQLAlchemy engine (resource, not pickled).
    Tuned for remote DB with limited max connections.
    """
    # Prefer Streamlit secrets for pool size if available
    try:
        pool_size_env = int(st.secrets.get("DB_POOL", os.getenv("DB_POOL", 10)))
    except Exception:
        pool_size_env = int(os.getenv("DB_POOL", 10))
    return create_engine(
        uri,
        pool_pre_ping=True,
        pool_size=pool_size_env,  # configurable pool size (default 10)
        max_overflow=0,     # do not exceed pool size
        pool_recycle=1800,  # recycle every 30 minutes
    )


@st.cache_data(show_spinner=False, ttl=300)
def fetch_many(engine_uri: str, queries: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Execute multiple queries using a single connection, return dict of DataFrames.
    Cached to minimize remote DB connections and repeated queries within TTL.
    """
    out: dict[str, pd.DataFrame] = {}
    try:
        eng = get_engine(engine_uri)
        with eng.connect() as conn:
            for key, q in queries.items():
                try:
                    out[key] = pd.read_sql(text(q), conn)
                except Exception as qe:
                    st.error(f"Query '{key}' failed: {qe}")
                    out[key] = pd.DataFrame()
    except Exception as e:
        st.error(f"Database error (connection): {e}")
        for key in queries.keys():
            out[key] = pd.DataFrame()
    return out


def to_dt(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def apply_sort(df: pd.DataFrame, sort_col: Optional[str], ascending: bool) -> pd.DataFrame:
    if sort_col and sort_col in df.columns:
        return df.sort_values(by=sort_col, ascending=ascending)
    return df


def reset_keys(keys: list[str]):
    """Utility to clear a list of st.session_state keys if they exist."""
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


# ---------------
# App UI
# ---------------
st.set_page_config(page_title="ISPSC Tagudin DMS Analytics", layout="wide")

# Custom CSS for clean dashboard layout
st.markdown("""
<style>
    /* Main container */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; color: #6B7280; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #f8f9fa;
        border-right: 1px solid #e5e7eb;
        padding: 0.75rem 0.75rem; /* better breathing room */
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { 
        color: #1f2937;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 0.25rem;
    }
    section[data-testid="stSidebar"] hr { border-color: #e5e7eb; }
    
    /* Improve spacing of controls */
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stMultiSelect,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stDateInput {
        margin-bottom: 0.5rem;
    }
    
    /* Accessible focus outlines */
    section[data-testid="stSidebar"] *:focus {
        outline: 2px solid #2563eb !important; /* blue focus */
        outline-offset: 2px;
        border-radius: 6px;
    }
    
    /* Improve clickable labels */
    section[data-testid="stSidebar"] label { cursor: pointer; }
    
    /* Minimal dot icon */
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    .dot-primary { background-color: #2563eb; }
    
    /* Header */
    .dashboard-header {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .dashboard-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .dashboard-subtitle { color: #6B7280; font-size: 0.95rem; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='color: #1f2937; text-align: center; margin-bottom: 0;'><span class='dot dot-primary'></span> ISPSC DMS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; text-align: center; font-size: 0.85rem; margin-top: 0.25rem;'>Document Management System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Dashboard", "Documents Analytics", "Documents Table"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Year Filter")
    
    # Get available years from data (will be populated after data fetch)
    current_year = datetime.now().year
    available_years = list(range(current_year - 5, current_year + 1))
    selected_year = st.selectbox(
        "Select Year",
        options=["All Years"] + available_years,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Department Filter")
    
    # Department filter options (multiselect)
    department_options = ["CAS", "CMBE", "CTE", "LHS", "Graduate School", "Others"]
    selected_departments = st.multiselect(
        "Select Departments",
        options=department_options,
        default=[],
        label_visibility="collapsed",
        placeholder="Select departments..."
    )
    
    st.markdown("---")
    st.markdown("### Folder Filter")
    
    # Folder filter (multiselect - will be populated dynamically)
    selected_folders = st.multiselect(
        "Select Folders",
        options=st.session_state.get("folder_filter_options", []),
        default=[],
        label_visibility="collapsed",
        placeholder="Select folders...",
        key="folder_filter"
    )
    
    st.markdown("---")
    st.markdown("### Document Type Filter")
    
    # Document type filter (multiselect)
    selected_doc_types = st.multiselect(
        "Select Document Types",
        options=st.session_state.get("doctype_filter_options", []),
        default=[],
        label_visibility="collapsed",
        placeholder="Select document types...",
        key="doctype_filter"
    )
    
    st.markdown("---")
    st.markdown("### Contributor Filter")
    
    # Contributor filter (multiselect)
    selected_contributors = st.multiselect(
        "Select Contributors",
        options=st.session_state.get("contrib_filter_options", []),
        default=[],
        label_visibility="collapsed",
        placeholder="Select contributors...",
        key="contrib_filter"
    )
    
    st.markdown("---")
    st.markdown("### Status Filter")
    
    # Status filter
    status_options = ["All Status", "Active", "Deleted"]
    selected_status = st.selectbox(
        "Select Status",
        options=status_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Date Range Filter")
    
    # Date range filter
    use_date_range = st.checkbox("Enable date range filter", value=False)
    if use_date_range:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("From", value=None, key="start_date")
        with col_d2:
            end_date = st.date_input("To", value=None, key="end_date")
    else:
        start_date = None
        end_date = None
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%b %d, %Y %H:%M')}")

# Build DB connection from environment variables (no sidebar controls)
# Defaults set to provided remote SQL details; prefer st.secrets at runtime
def cfg(key: str, default: str) -> str:
    try:
        return str(st.secrets.get(key, os.getenv(key, default)))
    except Exception:
        return str(os.getenv(key, default))

host = cfg("DB_HOST", "srv2050.hstgr.io")
port = int(cfg("DB_PORT", 3306))
user = cfg("DB_USER", "u185173985_Ladera")
# Prefer DB_PASS, fallback to DB_PASSWORD if present
password = cfg("DB_PASS", cfg("DB_PASSWORD", "Ladera102030"))
database = cfg("DB_NAME", "u185173985_ispsc_tag_dms")
st.session_state.engine_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

engine_uri = st.session_state.engine_uri
engine = get_engine(engine_uri)

# Centralize queries and fetch in one connection
docs_q = """
    SELECT d.doc_id,
           d.doc_type,
           t.name AS doc_type_name,
           d.folder_id,
           f.name AS folder_name,
           d.reference,
           d.title,
           d.status,
           d.deleted,
           d.available_copy,
           d.visibility,
           d.date_received,
           d.created_at,
           d.updated_at,
           d.created_by_name,
           GROUP_CONCAT(DISTINCT dept.code SEPARATOR ', ') AS department_codes,
           GROUP_CONCAT(DISTINCT dept.name SEPARATOR ', ') AS department_names
    FROM dms_documents d
    LEFT JOIN folders f ON d.folder_id = f.folder_id
    LEFT JOIN document_types t ON d.doc_type = t.type_id
    LEFT JOIN document_departments dd ON d.doc_id = dd.doc_id
    LEFT JOIN departments dept ON dd.department_id = dept.department_id
    GROUP BY d.doc_id
"""
requests_q = """
    SELECT da.document_action_id,
           da.doc_id,
           d.title AS document_title,
           da.action_id,
           ar.action_name,
           da.assigned_to_user_id,
           da.assigned_to_role,
           da.assigned_to_department_id,
           da.status,
           da.priority,
           da.due_date,
           da.completed_at,
           da.created_at,
           da.updated_at
    FROM document_actions da
    LEFT JOIN dms_documents d ON d.doc_id = da.doc_id
    LEFT JOIN action_required ar ON ar.action_id = da.action_id
"""
data = fetch_many(engine_uri, {"docs": docs_q, "reqs": requests_q})

# Global chart style
# 20-color, colorblind-friendly qualitative palette (mix of Plotly + Tableau)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
    "#ff9da7", "#9c755f", "#bab0ab", "#76b7b2", "#edc948"
]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

# Apply year filter to data
docs_df_full = data["docs"].copy()
reqs_df_full = data["reqs"].copy()

if not docs_df_full.empty:
    docs_df_full = to_dt(docs_df_full, ["date_received", "created_at", "updated_at"])
if not reqs_df_full.empty:
    reqs_df_full = to_dt(reqs_df_full, ["due_date", "completed_at", "created_at", "updated_at"])

# Filter by year
if selected_year != "All Years":
    if not docs_df_full.empty and "date_received" in docs_df_full.columns:
        docs_df_full = docs_df_full[
            docs_df_full["date_received"].dt.year == selected_year
        ]
    if not reqs_df_full.empty and "created_at" in reqs_df_full.columns:
        reqs_df_full = reqs_df_full[
            reqs_df_full["created_at"].dt.year == selected_year
        ]

# Populate dynamic filter options from data
if not docs_df_full.empty:
    # Get unique folders
    folder_options = sorted(docs_df_full["folder_name"].dropna().unique().tolist())
    # Get unique document types
    doc_type_options = sorted(docs_df_full["doc_type_name"].dropna().unique().tolist())
    # Get unique contributors
    contrib_options = (
        sorted(docs_df_full["created_by_name"].dropna().astype(str).unique().tolist())
        if "created_by_name" in docs_df_full.columns else []
    )
    
    # Update multiselect options using session state
    rerun_needed = False
    # Update options in session state and trigger rerun if they changed
    if folder_options:
        if st.session_state.get("folder_filter_options") != folder_options:
            st.session_state.folder_filter_options = folder_options
            rerun_needed = True
    if doc_type_options:
        if st.session_state.get("doctype_filter_options") != doc_type_options:
            st.session_state.doctype_filter_options = doc_type_options
            rerun_needed = True
    if contrib_options:
        if st.session_state.get("contrib_filter_options") != contrib_options:
            st.session_state.contrib_filter_options = contrib_options
            rerun_needed = True
    if rerun_needed:
        st.rerun()
else:
    folder_options = []
    doc_type_options = []
    contrib_options = []

# Filter by departments (multiselect)
if selected_departments and not docs_df_full.empty and "department_codes" in docs_df_full.columns:
    dept_mask = pd.Series([False] * len(docs_df_full), index=docs_df_full.index)
    
    for dept in selected_departments:
        if dept == "Others":
            # Include user-specific documents
            dept_mask |= (
                (docs_df_full["visibility"].isin(["SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"])) |
                (docs_df_full["department_codes"].isna())
            )
        else:
            # Include documents with this department code
            dept_mask |= (
                (docs_df_full["department_codes"].notna()) &
                (docs_df_full["department_codes"].str.contains(dept, na=False))
            )
    
    docs_df_full = docs_df_full[dept_mask]

# Filter by folders (multiselect)
if selected_folders and not docs_df_full.empty and "folder_name" in docs_df_full.columns:
    docs_df_full = docs_df_full[docs_df_full["folder_name"].isin(selected_folders)]

# Filter by document types (multiselect)
if selected_doc_types and not docs_df_full.empty and "doc_type_name" in docs_df_full.columns:
    docs_df_full = docs_df_full[docs_df_full["doc_type_name"].isin(selected_doc_types)]

# Filter by contributors (multiselect)
if 'selected_contributors' in locals() and selected_contributors and not docs_df_full.empty and "created_by_name" in docs_df_full.columns:
    docs_df_full = docs_df_full[docs_df_full["created_by_name"].isin(selected_contributors)]

# Filter by status
if selected_status != "All Status":
    if not docs_df_full.empty and "status" in docs_df_full.columns:
        status_value = selected_status.lower()
        docs_df_full = docs_df_full[docs_df_full["status"] == status_value]

# Filter by date range
if use_date_range and start_date and end_date:
    if not docs_df_full.empty and "date_received" in docs_df_full.columns:
        docs_df_full = docs_df_full[
            (docs_df_full["date_received"].dt.date >= start_date) &
            (docs_df_full["date_received"].dt.date <= end_date)
        ]

# Page routing based on sidebar selection
if page == "Dashboard":
    # Dashboard Overview Page
    st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="dashboard-title">Document Management Analytics</div>', unsafe_allow_html=True)
        year_display = selected_year if selected_year != "All Years" else "All Time"
        dept_display = ", ".join(selected_departments) if selected_departments else "All Departments"
        st.markdown(f'<div class="dashboard-subtitle">{year_display} - {dept_display}</div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>Total Documents<br/><span style='font-size: 1.5rem; font-weight: 600; color: #111827;'>{len(docs_df_full):,}</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    docs_df = docs_df_full.copy()
    
    # Top row - Key metrics (6 cards)
    st.markdown("#### Key Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    total_docs = len(docs_df)
    active_docs = (docs_df['status'] == 'active').sum() if not docs_df.empty else 0
    deleted_docs = (docs_df['status'] == 'deleted').sum() if not docs_df.empty else 0
    folder_count = docs_df["folder_name"].nunique() if not docs_df.empty and "folder_name" in docs_df.columns else 0
    doc_type_count = docs_df["doc_type_name"].nunique() if not docs_df.empty and "doc_type_name" in docs_df.columns else 0
    active_pct = (active_docs / total_docs * 100) if total_docs > 0 else 0
    
    with m1:
        st.metric("Total Documents", f"{total_docs:,}", help="All documents in the system")
    with m2:
        st.metric("Active Docs", f"{active_docs:,}", help="Currently active documents")
    with m3:
        st.metric("Permanently Deleted Docs", f"{deleted_docs:,}", help="Deleted documents")
    with m4:
        st.metric("Active Rate", f"{active_pct:.1f}%", help="Percentage of active documents")
    with m5:
        st.metric("Folders", f"{folder_count:,}", help="Number of folders")
    with m6:
        st.metric("Doc Types", f"{doc_type_count:,}", help="Number of document types")
    
    st.markdown("---")
    
    # Second row - Charts (line chart + donut chart like the image)
    chart_col1, chart_col2 = st.columns([1.5, 1])
    
    with chart_col1:
        st.markdown("**Document Intake Trend**")
        if not docs_df.empty and "date_received" in docs_df.columns:
            ts_data = docs_df.dropna(subset=["date_received"]).copy()
            ts_data["date"] = pd.to_datetime(ts_data["date_received"]).dt.to_period("W").dt.to_timestamp()
            ts = ts_data.groupby("date").size().reset_index(name="Documents")
            fig_line = px.line(ts, x="date", y="Documents", markers=True)
            fig_line.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No date data available")
    
    with chart_col2:
        st.markdown("**Document Status Distribution**")
        if not docs_df.empty and "status" in docs_df.columns:
            # Compute Deleted via explicit 'deleted' flag if available, else via status=='deleted'
            if "deleted" in docs_df.columns:
                deleted_mask = docs_df["deleted"].fillna(0).astype(int) == 1
            else:
                deleted_mask = docs_df["status"].astype(str).str.lower() == "deleted"

            deleted_count = int(deleted_mask.sum())
            active_count = int(len(docs_df) - deleted_count)
            status_counts = pd.DataFrame({
                "Status": ["Active", "Deleted"],
                "Count": [active_count, deleted_count],
            })

            fig_status_bar = px.bar(
                status_counts,
                x="Count",
                y="Status",
                orientation="h",
                color="Status",
                color_discrete_map={"Active": "#22c55e", "Deleted": "#ef4444"},
            )
            fig_status_bar.update_traces(texttemplate='%{x}', textposition='outside', hovertemplate='<b>%{y}</b><br>Documents: %{x}<extra></extra>', marker_line_color='#111', marker_line_width=0.5)
            fig_status_bar.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig_status_bar, use_container_width=True)
        else:
            st.info("No status data available")
    
    st.markdown("---")
    
    # Third row - Department Assignment Chart
    st.markdown("---")
    st.markdown("**Document Assignment by Department**")
    
    if not docs_df.empty and "visibility" in docs_df.columns:
        # Map visibility and department_codes to specific departments
        def map_to_department(row):
            vis = row.get("visibility", "")
            dept_codes = row.get("department_codes", "")
            
            if vis == "ALL":
                return "ALL"
            elif vis == "DEPARTMENT" and pd.notna(dept_codes) and dept_codes:
                # Return the actual department code(s)
                return dept_codes
            elif vis in ["SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"]:
                return "Others (User Specific)"
            else:
                return "Unassigned"
        
        docs_df["department"] = docs_df.apply(map_to_department, axis=1)
        dept_counts = docs_df["department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        
        # Add caption
        all_count = dept_counts[dept_counts["Department"] == "ALL"]["Count"].sum() if "ALL" in dept_counts["Department"].values else 0
        st.caption(f"{all_count} documents accessible to all departments")
        
        fig_dept = px.bar(dept_counts, x="Department", y="Count", color_discrete_sequence=PALETTE)
        fig_dept.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_dept, use_container_width=True)
    else:
        st.info("No department data available")

    # Contributors section
    st.markdown("---")
    st.markdown("**Top Contributors**")
    if not docs_df.empty and "created_by_name" in docs_df.columns:
        contrib_counts = (
            docs_df["created_by_name"].fillna("(Unknown)").value_counts().head(10).reset_index()
        )
        contrib_counts.columns = ["Contributor", "Documents"]
        fig_contrib = px.bar(
            contrib_counts,
            x="Documents",
            y="Contributor",
            orientation="h",
            color_discrete_sequence=PALETTE,
        )
        fig_contrib.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0), yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_contrib, use_container_width=True)
    else:
        st.info("No contributor data available")

    st.markdown("---")
    st.markdown("**Recent Documents**")
    if not docs_df.empty:
        # Choose best available date for recency
        date_col = "date_received" if "date_received" in docs_df.columns else ("created_at" if "created_at" in docs_df.columns else None)
        rec_df = docs_df.copy()
        if date_col:
            rec_df = rec_df.dropna(subset=[date_col])
            rec_df = rec_df.sort_values(date_col, ascending=False)
        cols = [c for c in ["doc_id", "title", "doc_type_name", "status", "created_by_name", "date_received", "created_at"] if c in rec_df.columns]
        if cols:
            st.dataframe(rec_df[cols].head(15), use_container_width=True, hide_index=True)
        else:
            st.info("No displayable columns for recent documents")

elif page == "Documents Analytics":
    # Header
    year_display = selected_year if selected_year != "All Years" else "All Time"
    dept_display = ", ".join(selected_departments) if selected_departments else "All Departments"
    st.markdown(f"### Document Analytics - {year_display} - {dept_display}")
    st.markdown("*Visual insights into your document ecosystem: status trends, distribution patterns, and intake analysis.*")
    
    docs_df = docs_df_full.copy()
    if not docs_df.empty: 

        # Calculate insights
        total_docs = len(docs_df)
        active_docs = (docs_df['status'] == 'active').sum()
        deleted_docs = (docs_df.get('deleted', 0) == 1).sum()
        active_pct = (active_docs / total_docs * 100) if total_docs > 0 else 0
        
        # KPI cards with storytelling
        d_k1, d_k2, d_k3 = st.columns(3)
        with d_k1:
            st.metric("Total Documents", f"{total_docs:,}")
            st.caption("Your complete document repository")
        with d_k2:
            st.metric("Active Documents", f"{active_docs:,}", delta=f"{active_pct:.1f}% of total")
            st.caption("Currently accessible and in use")
        with d_k3:
            st.metric("Deleted Documents", f"{deleted_docs:,}")
            st.caption("Archived or removed from circulation")
        
        # Insight banner
        if active_pct > 80:
            st.success(f"Healthy System: {active_pct:.1f}% of your documents are active, indicating good document lifecycle management.")
        elif active_pct > 50:
            st.info(f"Moderate Activity: {active_pct:.1f}% active documents. Consider reviewing document retention policies.")
        else:
            st.warning(f"Low Activity: Only {active_pct:.1f}% documents are active. Review your document retention policy.")
        
        st.markdown("---")
        
        # Charts only - no tables or filters
        f_docs = docs_df.copy()
        
        # Row 1: Status & Timeline
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Status Distribution**")
            s_counts = f_docs["status"].value_counts().reset_index()
            s_counts.columns = ["status", "Count"]
            top_status = s_counts.iloc[0]["status"] if not s_counts.empty else "N/A"
            top_count = s_counts.iloc[0]["Count"] if not s_counts.empty else 0
            st.caption(f"Most common: {top_status} ({top_count} documents)")
            fig = px.bar(s_counts, x="Count", y="status", orientation="h", color="status", color_discrete_sequence=PALETTE)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("**Document Intake Over Time**")
            if "date_received" in f_docs.columns and not f_docs.empty:
                ts_data = f_docs.dropna(subset=["date_received"]).copy()
                ts_data["month"] = pd.to_datetime(ts_data["date_received"]).dt.to_period("M").dt.to_timestamp()
                ts = ts_data.groupby("month").size().reset_index(name="Documents")
                
                if len(ts) >= 2:
                    recent_avg = ts.tail(3)["Documents"].mean()
                    older_avg = ts.head(3)["Documents"].mean()
                    if recent_avg > older_avg * 1.2:
                        st.caption("Trend: Intake is increasing")
                    elif recent_avg < older_avg * 0.8:
                        st.caption("Trend: Intake is decreasing")
                    else:
                        st.caption("Trend: Intake remains stable")
                
                fig = px.line(ts, x="month", y="Documents", markers=True)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No date data available")

        # Row 2: Folder & Doc Type
        st.markdown("---")
        c3, c4 = st.columns(2)
        
        with c3:
            if "folder_name" in f_docs.columns and not f_docs.empty:
                st.markdown("**Folder Distribution**")
                f_counts = f_docs["folder_name"].fillna("(No Folder)").value_counts().head(10).reset_index()
                f_counts.columns = ["folder_name", "Count"]
                total_folders = f_docs["folder_name"].nunique()
                st.caption(f"{total_folders} folders in use")
                fig = px.bar(f_counts, x="Count", y="folder_name", orientation="h", color_discrete_sequence=PALETTE)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with c4:
            if "doc_type_name" in f_docs.columns and not f_docs.empty:
                st.markdown("**Document Type Breakdown**")
                dt_counts = f_docs["doc_type_name"].fillna("(No Type)").value_counts().reset_index()
                dt_counts.columns = ["doc_type_name", "Count"]
                type_count = len(dt_counts)
                st.caption(f"{type_count} document types")
                fig = px.bar(dt_counts, x="Count", y="doc_type_name", orientation="h", color="doc_type_name", color_discrete_sequence=PALETTE)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Department Assignment
        if "visibility" in f_docs.columns and not f_docs.empty:
            st.markdown("---")
            st.markdown("**🏢 Document Assignment by Department**")
            
            # Map visibility and department_codes to specific departments
            def map_to_department(row):
                vis = row.get("visibility", "")
                dept_codes = row.get("department_codes", "")
                
                if vis == "ALL":
                    return "ALL"
                elif vis == "DEPARTMENT" and pd.notna(dept_codes) and dept_codes:
                    # Return the actual department code(s)
                    return dept_codes
                elif vis in ["SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"]:
                    return "Others (User Specific)"
                else:
                    return "Unassigned"
            
            f_docs["department"] = f_docs.apply(map_to_department, axis=1)
            dept_counts = f_docs["department"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Count"]
            
            all_count = dept_counts[dept_counts["Department"] == "ALL"]["Count"].sum() if "ALL" in dept_counts["Department"].values else 0
            st.caption(f"{all_count} documents accessible to all departments")
            
            fig = px.bar(dept_counts, x="Department", y="Count", color_discrete_sequence=PALETTE)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents found or failed to load documents.")

elif page == "Documents Table":
    # Header
    year_display = selected_year if selected_year != "All Years" else "All Time"
    dept_display = ", ".join(selected_departments) if selected_departments else "All Departments"
    st.markdown(f"### Documents Table - {year_display} - {dept_display}")
    st.markdown("*Complete list of documents with all details in tabular format.*")
    
    docs_df = docs_df_full.copy()
    
    if not docs_df.empty:
        # Show filter summary
        st.markdown(f"**Showing {len(docs_df):,} documents**")
        
        # Add search functionality
        search_term = st.text_input("Search documents (title, reference, from, to)", "")
        
        if search_term:
            search_cols = ["title", "reference", "from_field", "to_field"]
            mask = docs_df[search_cols].apply(
                lambda x: x.astype(str).str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            docs_df = docs_df[mask]
            st.caption(f"Found {len(docs_df):,} matching documents")
        
        # Select columns to display
        display_cols = [
            "doc_id", "reference", "title", "doc_type_name", "folder_name",
            "from_field", "to_field", "date_received", "status", 
            "department_codes", "available_copy", "created_at", "created_by_name"
        ]
        
        # Filter only existing columns
        display_cols = [col for col in display_cols if col in docs_df.columns]
        
        # Sort options
        st.markdown("---")
        col_sort1, col_sort2 = st.columns([3, 1])
        with col_sort1:
            sort_by = st.selectbox("Sort by", display_cols, index=display_cols.index("date_received") if "date_received" in display_cols else 0)
        with col_sort2:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
        
        # Apply sorting
        docs_df = docs_df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
        
        # Display table
        st.markdown("---")
        st.dataframe(
            docs_df[display_cols],
            use_container_width=True,
            hide_index=True,
            height=600
        )
        
        # Download button
        csv = docs_df[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No documents found matching the selected filters.")

