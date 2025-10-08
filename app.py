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


# ---------------
# App UI
# ---------------
st.set_page_config(page_title="ISPSC Tagudin DMS Analytics", page_icon="📊", layout="wide")

# Global styles (modern light theme)
st.markdown(
    """
    <style>
      :root {
        --primary: #4F46E5; /* indigo-600 */
        --bg: #F7F8FA;
        --panel: #FFFFFF;
        --muted: #6B7280;
        --shadow: 0 8px 20px rgba(0,0,0,0.06);
        --radius: 14px;
      }
      .stApp {
        background: var(--bg);
      }
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
      }
      /* Header */
      .app-header {
        background: var(--panel);
        border: 1px solid #EEF0F4;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
      }
      .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
      }
      .app-subtitle {
        color: var(--muted);
        margin-top: .25rem;
      }
      /* Tabs */
      [data-baseweb="tab-list"]{ gap: .25rem; }
      [data-baseweb="tab"]{ background: transparent; border-radius: 10px; }
      [data-baseweb="tab"]:hover{ background: #F1F5F9; }
      /* Cards/expanders */
      .stExpander, .stDataFrame, .element-container:has(.metric-container) {
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow);
        overflow: hidden;
        background: var(--panel);
        border: 1px solid #EEF0F4;
      }
      /* Buttons */
      .stButton>button {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        padding: .6rem .9rem;
        font-weight: 600;
      }
      .stButton>button[kind="primary"] {
        background: var(--primary);
        border-color: var(--primary);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with right-aligned refresh
header = st.container()
with header:
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    hcol1, hsp, hcol2 = st.columns([0.8, 0.05, 0.15])
    with hcol1:
        st.markdown('<div class="app-title">ISPSC Tagudin DMS Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Modern documents analytics dashboard</div>', unsafe_allow_html=True)
    with hcol2:
        if st.button("Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown('</div>', unsafe_allow_html=True)

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
           d.created_by_name
    FROM dms_documents d
    LEFT JOIN folders f ON d.folder_id = f.folder_id
    LEFT JOIN document_types t ON d.doc_type = t.type_id
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
PALETTE = [
    "#4F46E5", "#F59E0B", "#10B981", "#EC4899", "#0EA5E9",
    "#22D3EE", "#A78BFA", "#84CC16", "#F97316", "#14B8A6"
]
px.defaults.template = "plotly_white"

# Minimal layout: two tabs for Documents and Requests
docs_tab, req_tab = st.tabs(["Documents", "Requests"])

# -----------------------------
# Documents
# -----------------------------
with docs_tab:
    st.caption("Documents")
    docs_df = data["docs"].copy()
    if not docs_df.empty:
        docs_df = to_dt(docs_df, ["date_received", "created_at", "updated_at"]) 

        # KPI cards
        d_k1, d_k2, d_k3 = st.columns(3)
        with d_k1:
            st.metric("Total Documents", f"{len(docs_df):,}")
        with d_k2:
            st.metric("Active Status", f"{(docs_df['status'] == 'active').sum():,}")
        with d_k3:
            st.metric("Deleted Flag", f"{(docs_df.get('deleted', 0) == 1).sum():,}")

        dcol_f, dcol_s = st.columns(2)
        with dcol_f:
            with st.expander("Filters", expanded=False):
                exclude_deleted = st.checkbox("Exclude deleted", value=True, key="doc_exclude_deleted")
                statuses = sorted([s for s in docs_df["status"].dropna().unique()])
                status_sel = st.multiselect("Status", statuses, default=statuses, key="doc_status")
                vis_all_options = ["ALL", "DEPARTMENT", "SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"]
                vis_sel = st.multiselect("Visibility", vis_all_options, default=vis_all_options, key="doc_visibility")
                copy_vals = sorted([v for v in docs_df["available_copy"].dropna().unique()])
                copy_sel = st.multiselect("Available Copy", copy_vals, default=copy_vals, key="doc_copy")
                # Doc Type filter
                if "doc_type_name" in docs_df.columns:
                    dtype_vals = sorted([v for v in docs_df["doc_type_name"].dropna().unique()])
                    dtype_sel = st.multiselect("Doc Type", dtype_vals, default=dtype_vals, key="doc_dtype")
                else:
                    dtype_sel = None
                # Folder filter
                if "folder_name" in docs_df.columns:
                    folder_vals = sorted([v for v in docs_df["folder_name"].dropna().unique()])
                    folder_sel = st.multiselect("Folder", folder_vals, default=folder_vals, key="doc_folder")
                else:
                    folder_sel = None
                if "date_received" in docs_df.columns and not docs_df["date_received"].isna().all():
                    dmin = pd.to_datetime(docs_df["date_received"].min()).date()
                    dmax = pd.to_datetime(docs_df["date_received"].max()).date()
                    dstart, dend = st.date_input("Date received between", value=(dmin, dmax), key="doc_date_between")
                else:
                    dstart, dend = None, None

        f_docs = docs_df.copy()
        if status_sel:
            f_docs = f_docs[f_docs["status"].isin(status_sel)]
        if vis_sel:
            f_docs = f_docs[f_docs["visibility"].isin(vis_sel)]
        if copy_sel:
            f_docs = f_docs[f_docs["available_copy"].isin(copy_sel)]
        if dtype_sel is not None:
            f_docs = f_docs[f_docs["doc_type_name"].isin(dtype_sel)]
        if folder_sel is not None:
            f_docs = f_docs[f_docs["folder_name"].isin(folder_sel)]
        if exclude_deleted and "deleted" in f_docs.columns:
            f_docs = f_docs[(f_docs["deleted"].fillna(0) == 0)]
        if dstart and dend and "date_received" in f_docs.columns:
            f_docs = f_docs[(f_docs["date_received"].dt.date >= dstart) & (f_docs["date_received"].dt.date <= dend)]

        # Default sort for documents (no UI)
        sort_col_d = "doc_id"
        asc_d = True
        f_docs = apply_sort(f_docs, sort_col_d, asc_d)

        # Display with friendly columns: Folder Name and Doc Type
        display_cols = [
            c for c in [
                "doc_id",
                "title",
                "doc_type_name",
                "folder_name",
                "status",
                "visibility",
                "available_copy",
                "reference",
                "date_received",
                "created_at",
                "updated_at",
                "created_by_name",
            ] if c in f_docs.columns
        ]
        st.dataframe(f_docs[display_cols], use_container_width=True, hide_index=True)

        with st.expander("Chart toggles", expanded=False):
            show_d_status = st.checkbox("Documents by Status", value=True, key="d_status")
            d_status_type = st.selectbox("Documents by Status chart type", ["Bar", "Pie"], index=0, key="d_status_type")
            show_d_time = st.checkbox("Documents over Time", value=True, key="d_time")
            d_time_type = st.selectbox("Documents over Time chart type", ["Line", "Bar"], index=0, key="d_time_type")
            show_d_folder = st.checkbox("Documents by Folder", value=True, key="d_folder")
            d_folder_type = st.selectbox("Documents by Folder chart type", ["Bar", "Pie"], index=0, key="d_folder_type")
            show_d_dtype = st.checkbox("Documents by Doc Type", value=True, key="d_dtype")
            d_dtype_type = st.selectbox("Documents by Doc Type chart type", ["Bar", "Pie"], index=0, key="d_dtype_type")
            show_d_visibility = st.checkbox("Documents by Visibility", value=True, key="d_visibility")
            d_visibility_type = st.selectbox("Documents by Visibility chart type", ["Pie", "Bar"], index=0, key="d_visibility_type")
            granularity_map = {"Day": "D", "Week": "W", "Month": "M"}
            granularity = st.selectbox("Time granularity", list(granularity_map.keys()), index=0, key="doc_granularity")

        c3, c4 = st.columns(2)
        with c3:
            if show_d_status:
                st.caption("Documents by Status")
                s_counts = f_docs["status"].value_counts().reset_index()
                s_counts.columns = ["status", "Count"]
                if d_status_type == "Bar":
                    fig = px.bar(s_counts, x="status", y="Count", color_discrete_sequence=PALETTE)
                else:
                    fig = px.pie(s_counts, names="status", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            if show_d_time:
                st.caption("Documents over Time (date_received)")
                if "date_received" in f_docs.columns and not f_docs.empty:
                    period = granularity_map[granularity]
                    ts = (
                        f_docs.dropna(subset=["date_received"]) 
                             .groupby(f_docs["date_received"].dt.to_period(period)).size()
                             .reset_index(name="Count")
                    )
                    ts["date"] = ts["date_received"].dt.to_timestamp()
                    if d_time_type == "Line":
                        fig = px.line(ts, x="date", y="Count")
                    else:
                        fig = px.bar(ts, x="date", y="Count", color_discrete_sequence=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No date_received values to plot.")

        # Optional: Documents by Folder chart
        if show_d_folder and "folder_name" in f_docs.columns and not f_docs.empty:
            st.caption("Documents by Folder")
            f_counts = f_docs["folder_name"].fillna("(No Folder)").value_counts().reset_index()
            f_counts.columns = ["folder_name", "Count"]
            if d_folder_type == "Bar":
                fig = px.bar(f_counts, x="folder_name", y="Count", color_discrete_sequence=PALETTE)
            else:
                fig = px.pie(f_counts, names="folder_name", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

        # Optional: Documents by Doc Type chart
        if show_d_dtype and "doc_type_name" in f_docs.columns and not f_docs.empty:
            st.caption("Documents by Doc Type")
            dt_counts = f_docs["doc_type_name"].fillna("(No Type)").value_counts().reset_index()
            dt_counts.columns = ["doc_type_name", "Count"]
            if d_dtype_type == "Bar":
                fig = px.bar(dt_counts, x="doc_type_name", y="Count", color_discrete_sequence=PALETTE)
            else:
                fig = px.pie(dt_counts, names="doc_type_name", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

        # Documents by Visibility pie
        if show_d_visibility and "visibility" in f_docs.columns and not f_docs.empty:
            st.caption("Documents by Visibility")
            vis_all_options = ["ALL", "DEPARTMENT", "SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"]
            v_counts = f_docs["visibility"].value_counts().reindex(vis_all_options, fill_value=0).reset_index()
            v_counts.columns = ["visibility", "Count"]
            if d_visibility_type == "Pie":
                fig = px.pie(v_counts, names="visibility", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            else:
                fig = px.bar(v_counts, x="visibility", y="Count", color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents found or failed to load documents.")


# -----------------------------
# Requests
# -----------------------------
with req_tab:
    st.caption("Document Requests")
    reqs_df = data["reqs"].copy()
    if not reqs_df.empty:
        reqs_df = to_dt(reqs_df, ["due_date", "completed_at", "created_at", "updated_at"]) 

        # KPI cards
        r_k1, r_k2, r_k3 = st.columns(3)
        with r_k1:
            st.metric("Total Requests", f"{len(reqs_df):,}")
        with r_k2:
            st.metric("Pending", f"{(reqs_df['status'] == 'pending').sum():,}")
        with r_k3:
            st.metric("Completed", f"{(reqs_df['status'] == 'completed').sum():,}")

        rcol_f, rcol_s = st.columns(2)
        with rcol_f:
            with st.expander("Filters", expanded=False):
                r_statuses = ["pending", "in_progress", "completed", "cancelled"]
                r_status_sel = st.multiselect("Status", r_statuses, default=r_statuses, key="req_status")
                r_priorities = ["low", "medium", "high", "urgent"]
                r_priority_sel = st.multiselect("Priority", r_priorities, default=r_priorities, key="req_priority")
                r_roles = ["ADMIN", "DEAN", "FACULTY"]
                r_role_sel = st.multiselect("Assigned to Role", r_roles, default=r_roles, key="req_role")
                if "due_date" in reqs_df.columns and not reqs_df["due_date"].isna().all():
                    rmin = pd.to_datetime(reqs_df["due_date"].min()).date()
                    rmax = pd.to_datetime(reqs_df["due_date"].max()).date()
                    rstart, rend = st.date_input("Due date between", value=(rmin, rmax), key="req_due_between")
                else:
                    rstart, rend = None, None

        f_reqs = reqs_df.copy()
        if r_status_sel:
            f_reqs = f_reqs[f_reqs["status"].isin(r_status_sel)]
        if r_priority_sel:
            f_reqs = f_reqs[f_reqs["priority"].isin(r_priority_sel)]
        if r_role_sel:
            f_reqs = f_reqs[f_reqs["assigned_to_role"].isin(r_role_sel)]
        if rstart and rend and "due_date" in f_reqs.columns:
            f_reqs = f_reqs[(f_reqs["due_date"].dt.date >= rstart) & (f_reqs["due_date"].dt.date <= rend)]

        # Default sort and table
        f_reqs = apply_sort(f_reqs, "document_action_id", True)
        display_cols_r = [
            c for c in [
                "document_action_id",
                "doc_id",
                "document_title",
                "action_name",
                "assigned_to_role",
                "assigned_to_department_id",
                "status",
                "priority",
                "due_date",
                "created_at",
                "updated_at",
            ] if c in f_reqs.columns
        ]
        st.dataframe(f_reqs[display_cols_r], use_container_width=True, hide_index=True)
    else:
        st.info("No document requests found or failed to load requests.")
 

