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
st.set_page_config(page_title="ISPSC Tagudin DMS Analytics", layout="wide", page_icon="📊")

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
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
    }
    section[data-testid="stSidebar"] .css-1d391kg { color: white; }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: white; }
    
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
    st.markdown("<h2 style='color: white; text-align: center;'>📊 ISPSC DMS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #93C5FD; text-align: center; font-size: 0.85rem;'>Document Management System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📈 Dashboard", "📄 Documents", "📋 Requests"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
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
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]
px.defaults.template = "plotly_dark"

# Page routing based on sidebar selection
if page == "📈 Dashboard":
    # Dashboard Overview Page
    st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="dashboard-title">Document Management Analytics</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-subtitle">{datetime.now().strftime("%B %Y")}</div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>Total Records<br/><span style='font-size: 1.5rem; font-weight: 600; color: #111827;'>{len(data['docs']) + len(data['reqs']):,}</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    docs_df = data["docs"].copy()
    reqs_df = data["reqs"].copy()
    
    if not docs_df.empty:
        docs_df = to_dt(docs_df, ["date_received", "created_at", "updated_at"])
    if not reqs_df.empty:
        reqs_df = to_dt(reqs_df, ["due_date", "completed_at", "created_at", "updated_at"])
    
    # Top row - Key metrics (6 cards like the image)
    st.markdown("#### 📊 Key Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    total_docs = len(docs_df)
    active_docs = (docs_df['status'] == 'active').sum() if not docs_df.empty else 0
    total_reqs = len(reqs_df)
    completed_reqs = (reqs_df['status'] == 'completed').sum() if not reqs_df.empty else 0
    pending_reqs = (reqs_df['status'] == 'pending').sum() if not reqs_df.empty else 0
    completion_rate = (completed_reqs / total_reqs * 100) if total_reqs > 0 else 0
    
    with m1:
        st.metric("Total Documents", f"{total_docs:,}", help="All documents in the system")
    with m2:
        st.metric("Active Docs", f"{active_docs:,}", help="Currently active documents")
    with m3:
        st.metric("Completion Rate", f"{completion_rate:.1f}%", help="Request completion percentage")
    with m4:
        st.metric("Total Requests", f"{total_reqs:,}", help="All document requests")
    with m5:
        st.metric("Pending", f"{pending_reqs:,}", help="Requests awaiting action")
    with m6:
        st.metric("Completed", f"{completed_reqs:,}", help="Finished requests")
    
    st.markdown("---")
    
    # Second row - Charts (line chart + donut chart like the image)
    chart_col1, chart_col2 = st.columns([1.5, 1])
    
    with chart_col1:
        st.markdown("**📈 Document Intake Trend**")
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
        st.markdown("**📊 Request Status Distribution**")
        if not reqs_df.empty and "status" in reqs_df.columns:
            status_counts = reqs_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_donut = px.pie(status_counts, names="Status", values="Count", hole=0.5)
            fig_donut.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=True)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No status data available")
    
    st.markdown("---")
    
    # Third row - Additional metrics (6 more cards)
    st.markdown("#### 📋 Additional Insights")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    
    deleted_docs = (docs_df.get('deleted', 0) == 1).sum() if not docs_df.empty else 0
    folder_count = docs_df["folder_name"].nunique() if not docs_df.empty and "folder_name" in docs_df.columns else 0
    doc_type_count = docs_df["doc_type_name"].nunique() if not docs_df.empty and "doc_type_name" in docs_df.columns else 0
    in_progress = (reqs_df['status'] == 'in_progress').sum() if not reqs_df.empty else 0
    urgent_reqs = (reqs_df['priority'] == 'urgent').sum() if not reqs_df.empty and 'priority' in reqs_df.columns else 0
    
    # Calculate average processing time (mock for now)
    avg_time = "2.3 days"
    
    with a1:
        st.metric("Deleted Docs", f"{deleted_docs:,}")
    with a2:
        st.metric("Folders", f"{folder_count:,}")
    with a3:
        st.metric("Doc Types", f"{doc_type_count:,}")
    with a4:
        st.metric("In Progress", f"{in_progress:,}")
    with a5:
        st.metric("Urgent", f"{urgent_reqs:,}")
    with a6:
        st.metric("Avg. Time", avg_time)

elif page == "📄 Documents":
    st.markdown("### 📄 Document Management Overview")
    st.markdown("*Explore your document ecosystem: track status, monitor trends, and identify patterns in your document management system.*")
    
    docs_df = data["docs"].copy()
    if not docs_df.empty:
        docs_df = to_dt(docs_df, ["date_received", "created_at", "updated_at"]) 

        # Calculate insights
        total_docs = len(docs_df)
        active_docs = (docs_df['status'] == 'active').sum()
        deleted_docs = (docs_df.get('deleted', 0) == 1).sum()
        active_pct = (active_docs / total_docs * 100) if total_docs > 0 else 0
        
        # KPI cards with storytelling
        d_k1, d_k2, d_k3 = st.columns(3)
        with d_k1:
            st.metric("Total Documents", f"{total_docs:,}")
            st.caption("📊 Your complete document repository")
        with d_k2:
            st.metric("Active Documents", f"{active_docs:,}", delta=f"{active_pct:.1f}% of total")
            st.caption("✅ Currently accessible and in use")
        with d_k3:
            st.metric("Deleted Documents", f"{deleted_docs:,}")
            st.caption("🗑️ Archived or removed from circulation")
        
        # Insight banner
        if active_pct > 80:
            st.success(f"💡 **Healthy System**: {active_pct:.1f}% of your documents are active, indicating good document lifecycle management.")
        elif active_pct > 50:
            st.info(f"📌 **Moderate Activity**: {active_pct:.1f}% active documents. Consider reviewing inactive documents for archival.")
        else:
            st.warning(f"⚠️ **Low Activity**: Only {active_pct:.1f}% documents are active. Review your document retention policy.")

        dcol_f, dcol_s = st.columns(2)
        with dcol_f:
            with st.expander("Filters", expanded=False):
                # Search
                search_text = st.text_input("Search (title/reference/created by)", key="doc_search")

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

                # Sort controls
                sortable_cols = [c for c in [
                    "doc_id", "title", "status", "visibility", "available_copy",
                    "date_received", "created_at", "updated_at"
                ] if c in docs_df.columns]
                scol1, scol2 = st.columns([0.7, 0.3])
                with scol1:
                    sort_col_d = st.selectbox("Sort by", sortable_cols, index=0 if "doc_id" in sortable_cols else 0, key="doc_sort_col")
                with scol2:
                    asc_d = st.checkbox("Ascending", value=True, key="doc_sort_asc")

                # Quick actions
                q1, q2 = st.columns(2)
                with q1:
                    if st.button("Reset filters", key="doc_reset_btn"):
                        reset_keys([
                            "doc_exclude_deleted", "doc_status", "doc_visibility", "doc_copy",
                            "doc_dtype", "doc_folder", "doc_date_between", "doc_search",
                            "doc_sort_col", "doc_sort_asc"
                        ])
                        st.rerun()
                with q2:
                    if st.button("Select all", key="doc_select_all_btn"):
                        st.session_state["doc_status"] = statuses
                        st.session_state["doc_visibility"] = vis_all_options
                        st.session_state["doc_copy"] = copy_vals
                        if "doc_dtype" in st.session_state and 'dtype_vals' in locals():
                            st.session_state["doc_dtype"] = dtype_vals
                        if "doc_folder" in st.session_state and 'folder_vals' in locals():
                            st.session_state["doc_folder"] = folder_vals
                        st.rerun()

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
        # Text search across key columns
        if st.session_state.get("doc_search"):
            q = st.session_state["doc_search"].strip().lower()
            search_cols = [c for c in ["title", "reference", "created_by_name"] if c in f_docs.columns]
            if search_cols:
                mask = False
                for c in search_cols:
                    mask = (mask | f_docs[c].astype(str).str.lower().str.contains(q, na=False)) if isinstance(mask, pd.Series) else f_docs[c].astype(str).str.lower().str.contains(q, na=False)
                f_docs = f_docs[mask]

        # Sort for documents (from UI)
        f_docs = apply_sort(f_docs, sort_col_d, asc_d)

        # Display with friendly columns: Folder Name and Doc Type (hidden by default)
        with st.expander("Show data table", expanded=False):
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

        # Storytelling: Chart insights
        st.markdown("---")
        st.markdown("#### 📈 Visual Analytics & Trends")
        
        c3, c4 = st.columns(2)
        with c3:
            if show_d_status:
                st.markdown("**Status Distribution**")
                s_counts = f_docs["status"].value_counts().reset_index()
                s_counts.columns = ["status", "Count"]
                
                # Add insight
                top_status = s_counts.iloc[0]["status"] if not s_counts.empty else "N/A"
                top_count = s_counts.iloc[0]["Count"] if not s_counts.empty else 0
                st.caption(f"🔍 Most common status: **{top_status}** ({top_count} documents)")
                
                if d_status_type == "Bar":
                    fig = px.bar(s_counts, x="status", y="Count", color_discrete_sequence=PALETTE)
                else:
                    fig = px.pie(s_counts, names="status", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            if show_d_time:
                st.markdown("**Document Intake Timeline**")
                if "date_received" in f_docs.columns and not f_docs.empty:
                    period = granularity_map[granularity]
                    ts = (
                        f_docs.dropna(subset=["date_received"]) 
                             .groupby(f_docs["date_received"].dt.to_period(period)).size()
                             .reset_index(name="Count")
                    )
                    ts["date"] = ts["date_received"].dt.to_timestamp()
                    
                    # Add trend insight
                    if len(ts) >= 2:
                        recent_avg = ts.tail(3)["Count"].mean()
                        older_avg = ts.head(3)["Count"].mean()
                        if recent_avg > older_avg * 1.2:
                            st.caption("📈 **Trend**: Document intake is increasing over time")
                        elif recent_avg < older_avg * 0.8:
                            st.caption("📉 **Trend**: Document intake is decreasing")
                        else:
                            st.caption("➡️ **Trend**: Document intake remains stable")
                    
                    if d_time_type == "Line":
                        fig = px.line(ts, x="date", y="Count")
                    else:
                        fig = px.bar(ts, x="date", y="Count", color_discrete_sequence=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No date_received values to plot.")

        # Optional: Documents by Folder chart
        if show_d_folder and "folder_name" in f_docs.columns and not f_docs.empty:
            st.markdown("---")
            st.markdown("**📁 Folder Organization**")
            f_counts = f_docs["folder_name"].fillna("(No Folder)").value_counts().reset_index()
            f_counts.columns = ["folder_name", "Count"]
            
            # Folder insights
            total_folders = len(f_counts)
            top_folder = f_counts.iloc[0]["folder_name"] if not f_counts.empty else "N/A"
            top_folder_count = f_counts.iloc[0]["Count"] if not f_counts.empty else 0
            st.caption(f"📂 **{total_folders}** folders in use | Top folder: **{top_folder}** ({top_folder_count} docs)")
            
            if d_folder_type == "Bar":
                fig = px.bar(f_counts, x="folder_name", y="Count", color_discrete_sequence=PALETTE)
            else:
                fig = px.pie(f_counts, names="folder_name", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

        # Optional: Documents by Doc Type chart
        if show_d_dtype and "doc_type_name" in f_docs.columns and not f_docs.empty:
            st.markdown("---")
            st.markdown("**📋 Document Type Breakdown**")
            dt_counts = f_docs["doc_type_name"].fillna("(No Type)").value_counts().reset_index()
            dt_counts.columns = ["doc_type_name", "Count"]
            
            # Type diversity insight
            type_count = len(dt_counts)
            dominant_type = dt_counts.iloc[0]["doc_type_name"] if not dt_counts.empty else "N/A"
            dominant_pct = (dt_counts.iloc[0]["Count"] / dt_counts["Count"].sum() * 100) if not dt_counts.empty else 0
            
            st.caption(f"📑 **{type_count}** document types | Most common: **{dominant_type}** ({dominant_pct:.1f}% of all documents)")
            
            if d_dtype_type == "Bar":
                fig = px.bar(dt_counts, x="doc_type_name", y="Count", color_discrete_sequence=PALETTE)
            else:
                fig = px.pie(dt_counts, names="doc_type_name", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

        # Documents by Visibility pie
        if show_d_visibility and "visibility" in f_docs.columns and not f_docs.empty:
            st.markdown("---")
            st.markdown("**🔐 Access & Visibility Control**")
            vis_all_options = ["ALL", "DEPARTMENT", "SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"]
            v_counts = f_docs["visibility"].value_counts().reindex(vis_all_options, fill_value=0).reset_index()
            v_counts.columns = ["visibility", "Count"]
            
            # Visibility insight
            public_docs = v_counts[v_counts["visibility"] == "ALL"]["Count"].sum()
            restricted_docs = v_counts[v_counts["visibility"] != "ALL"]["Count"].sum()
            
            if public_docs > restricted_docs:
                st.caption(f"🌐 **Open Access**: {public_docs} documents are publicly visible, promoting transparency")
            else:
                st.caption(f"🔒 **Controlled Access**: {restricted_docs} documents have restricted visibility for security")
            
            if d_visibility_type == "Pie":
                fig = px.pie(v_counts, names="visibility", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            else:
                fig = px.bar(v_counts, x="visibility", y="Count", color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)
        
        # Actionable Summary
        st.markdown("---")
        st.markdown("### 💡 Key Takeaways & Recommendations")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📌 What the data tells us:**")
            insights = []
            if active_pct > 80:
                insights.append("✓ Strong document lifecycle management")
            if "folder_name" in f_docs.columns:
                folder_diversity = f_docs["folder_name"].nunique()
                if folder_diversity > 5:
                    insights.append(f"✓ Well-organized with {folder_diversity} folders")
            if "doc_type_name" in f_docs.columns:
                type_diversity = f_docs["doc_type_name"].nunique()
                if type_diversity > 3:
                    insights.append(f"✓ Diverse document types ({type_diversity} categories)")
            
            for insight in insights:
                st.markdown(f"- {insight}")
        
        with col_b:
            st.markdown("**🎯 Suggested Actions:**")
            actions = []
            if deleted_docs > total_docs * 0.2:
                actions.append("Review and permanently archive old deleted documents")
            if active_pct < 50:
                actions.append("Audit inactive documents for potential archival")
            if restricted_docs < public_docs * 0.1:
                actions.append("Consider adding access controls for sensitive documents")
            
            if not actions:
                actions.append("System is well-maintained! Continue current practices")
            
            for action in actions:
                st.markdown(f"- {action}")
    else:
        st.info("No documents found or failed to load documents.")


elif page == "📋 Requests":
    st.markdown("### 📋 Document Request Workflow")
    st.markdown("*Track action items, monitor completion rates, and identify bottlenecks in your document request pipeline.*")
    
    reqs_df = data["reqs"].copy()
    if not reqs_df.empty:
        reqs_df = to_dt(reqs_df, ["due_date", "completed_at", "created_at", "updated_at"]) 

        # Calculate insights
        total_reqs = len(reqs_df)
        pending_reqs = (reqs_df['status'] == 'pending').sum()
        completed_reqs = (reqs_df['status'] == 'completed').sum()
        in_progress_reqs = (reqs_df['status'] == 'in_progress').sum()
        completion_rate = (completed_reqs / total_reqs * 100) if total_reqs > 0 else 0
        
        # KPI cards with storytelling
        r_k1, r_k2, r_k3, r_k4 = st.columns(4)
        with r_k1:
            st.metric("Total Requests", f"{total_reqs:,}")
            st.caption("📊 All document action items")
        with r_k2:
            st.metric("Pending", f"{pending_reqs:,}")
            st.caption("⏳ Awaiting action")
        with r_k3:
            st.metric("In Progress", f"{in_progress_reqs:,}")
            st.caption("🔄 Currently being worked on")
        with r_k4:
            st.metric("Completed", f"{completed_reqs:,}", delta=f"{completion_rate:.1f}% done")
            st.caption("✅ Successfully finished")
        
        # Performance insight
        if completion_rate >= 70:
            st.success(f"🎯 **Excellent Performance**: {completion_rate:.1f}% completion rate shows strong workflow efficiency!")
        elif completion_rate >= 50:
            st.info(f"📊 **Good Progress**: {completion_rate:.1f}% completed. Keep the momentum going!")
        elif completion_rate >= 30:
            st.warning(f"⚠️ **Needs Attention**: Only {completion_rate:.1f}% completed. Consider reviewing pending requests.")
        else:
            st.error(f"🚨 **Action Required**: {completion_rate:.1f}% completion rate is low. Immediate attention needed for pending items.")

        rcol_f, rcol_s = st.columns(2)
        with rcol_f:
            with st.expander("Filters", expanded=False):
                # Search
                r_search = st.text_input("Search (title/action/role)", key="req_search")

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

                # Sort controls
                r_sortable = [c for c in [
                    "document_action_id", "doc_id", "action_name", "assigned_to_role",
                    "status", "priority", "due_date", "created_at", "updated_at"
                ] if c in reqs_df.columns]
                rs1, rs2 = st.columns([0.7, 0.3])
                with rs1:
                    r_sort_col = st.selectbox("Sort by", r_sortable, index=0, key="req_sort_col")
                with rs2:
                    r_sort_asc = st.checkbox("Ascending", value=True, key="req_sort_asc")

                # Quick actions
                rq1, rq2 = st.columns(2)
                with rq1:
                    if st.button("Reset filters", key="req_reset_btn"):
                        reset_keys([
                            "req_status", "req_priority", "req_role", "req_due_between",
                            "req_search", "req_sort_col", "req_sort_asc"
                        ])
                        st.rerun()
                with rq2:
                    if st.button("Select all", key="req_select_all_btn"):
                        st.session_state["req_status"] = r_statuses
                        st.session_state["req_priority"] = r_priorities
                        st.session_state["req_role"] = r_roles
                        st.rerun()

        f_reqs = reqs_df.copy()
        if r_status_sel:
            f_reqs = f_reqs[f_reqs["status"].isin(r_status_sel)]
        if r_priority_sel:
            f_reqs = f_reqs[f_reqs["priority"].isin(r_priority_sel)]
        if r_role_sel:
            f_reqs = f_reqs[f_reqs["assigned_to_role"].isin(r_role_sel)]
        if rstart and rend and "due_date" in f_reqs.columns:
            f_reqs = f_reqs[(f_reqs["due_date"].dt.date >= rstart) & (f_reqs["due_date"].dt.date <= rend)]
        # Text search
        if st.session_state.get("req_search"):
            q2 = st.session_state["req_search"].strip().lower()
            r_search_cols = [c for c in ["document_title", "action_name", "assigned_to_role"] if c in f_reqs.columns]
            if r_search_cols:
                rmask = False
                for c in r_search_cols:
                    rmask = (rmask | f_reqs[c].astype(str).str.lower().str.contains(q2, na=False)) if isinstance(rmask, pd.Series) else f_reqs[c].astype(str).str.lower().str.contains(q2, na=False)
                f_reqs = f_reqs[rmask]

        # Sort and table (hidden by default)
        f_reqs = apply_sort(f_reqs, r_sort_col, r_sort_asc)
        with st.expander("Show data table", expanded=False):
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
        
        # Priority and workload insights
        st.markdown("---")
        st.markdown("### 📊 Workload Analysis")
        
        if "priority" in f_reqs.columns and not f_reqs.empty:
            priority_counts = f_reqs["priority"].value_counts()
            urgent_count = priority_counts.get("urgent", 0)
            high_count = priority_counts.get("high", 0)
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**⚡ Priority Distribution**")
                if urgent_count > 0:
                    st.warning(f"🚨 **{urgent_count}** urgent requests require immediate attention")
                if high_count > 0:
                    st.info(f"⚠️ **{high_count}** high-priority requests in queue")
                if urgent_count == 0 and high_count == 0:
                    st.success("✅ No urgent or high-priority requests pending")
            
            with col_p2:
                st.markdown("**👥 Role Distribution**")
                if "assigned_to_role" in f_reqs.columns:
                    role_counts = f_reqs["assigned_to_role"].value_counts()
                    busiest_role = role_counts.index[0] if len(role_counts) > 0 else "N/A"
                    busiest_count = role_counts.iloc[0] if len(role_counts) > 0 else 0
                    st.caption(f"Most assigned: **{busiest_role}** ({busiest_count} requests)")
                    
                    # Workload balance check
                    if len(role_counts) > 1:
                        max_load = role_counts.max()
                        min_load = role_counts.min()
                        if max_load > min_load * 3:
                            st.warning("⚖️ Workload imbalance detected - consider redistributing tasks")
        
        # Actionable Summary for Requests
        st.markdown("---")
        st.markdown("### 💡 Workflow Insights & Next Steps")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**📈 Current Status:**")
            req_insights = []
            if completion_rate >= 70:
                req_insights.append("✓ Excellent completion rate")
            if pending_reqs > 0:
                req_insights.append(f"⏳ {pending_reqs} requests awaiting action")
            if in_progress_reqs > 0:
                req_insights.append(f"🔄 {in_progress_reqs} requests in progress")
            
            for insight in req_insights:
                st.markdown(f"- {insight}")
        
        with col_r2:
            st.markdown("**🎯 Recommended Actions:**")
            req_actions = []
            if pending_reqs > completed_reqs:
                req_actions.append("Prioritize clearing pending backlog")
            if urgent_count > 0:
                req_actions.append(f"Address {urgent_count} urgent requests first")
            if completion_rate < 50:
                req_actions.append("Review workflow bottlenecks and resource allocation")
            
            if not req_actions:
                req_actions.append("Workflow is healthy! Maintain current pace")
            
            for action in req_actions:
                st.markdown(f"- {action}")
    else:
        st.info("No document requests found or failed to load requests.")
 

