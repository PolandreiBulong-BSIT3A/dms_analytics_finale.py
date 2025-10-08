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
        background: linear-gradient(180deg, #065f46 0%, #047857 100%);
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
    st.markdown("<p style='color: #86efac; text-align: center; font-size: 0.85rem;'>Document Management System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📈 Dashboard", "📄 Documents Analytics", "📋 Requests Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📅 Year Filter")
    
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

# Page routing based on sidebar selection
if page == "📈 Dashboard":
    # Dashboard Overview Page
    st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="dashboard-title">Document Management Analytics</div>', unsafe_allow_html=True)
        year_display = selected_year if selected_year != "All Years" else "All Time"
        st.markdown(f'<div class="dashboard-subtitle">{year_display} Overview</div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>Total Records<br/><span style='font-size: 1.5rem; font-weight: 600; color: #111827;'>{len(docs_df_full) + len(reqs_df_full):,}</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    docs_df = docs_df_full.copy()
    reqs_df = reqs_df_full.copy()
    
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

elif page == "📄 Documents Analytics":
    # Header
    year_display = selected_year if selected_year != "All Years" else "All Time"
    st.markdown(f"### 📄 Document Analytics - {year_display}")
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
        
        st.markdown("---")
        
        # Charts only - no tables or filters
        f_docs = docs_df.copy()
        
        # Row 1: Status & Timeline
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Status Distribution**")
            s_counts = f_docs["status"].value_counts().reset_index()
            s_counts.columns = ["status", "Count"]
            top_status = s_counts.iloc[0]["status"] if not s_counts.empty else "N/A"
            top_count = s_counts.iloc[0]["Count"] if not s_counts.empty else 0
            st.caption(f"🔍 Most common: **{top_status}** ({top_count} documents)")
            fig = px.pie(s_counts, names="status", values="Count", hole=0.4, color_discrete_sequence=PALETTE)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("**📈 Document Intake Over Time**")
            if "date_received" in f_docs.columns and not f_docs.empty:
                ts_data = f_docs.dropna(subset=["date_received"]).copy()
                ts_data["month"] = pd.to_datetime(ts_data["date_received"]).dt.to_period("M").dt.to_timestamp()
                ts = ts_data.groupby("month").size().reset_index(name="Documents")
                
                if len(ts) >= 2:
                    recent_avg = ts.tail(3)["Documents"].mean()
                    older_avg = ts.head(3)["Documents"].mean()
                    if recent_avg > older_avg * 1.2:
                        st.caption("📈 **Trend**: Intake is increasing")
                    elif recent_avg < older_avg * 0.8:
                        st.caption("📉 **Trend**: Intake is decreasing")
                    else:
                        st.caption("➡️ **Trend**: Intake remains stable")
                
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
                st.markdown("**📁 Folder Distribution**")
                f_counts = f_docs["folder_name"].fillna("(No Folder)").value_counts().head(10).reset_index()
                f_counts.columns = ["folder_name", "Count"]
                total_folders = f_docs["folder_name"].nunique()
                st.caption(f"📂 **{total_folders}** folders in use")
                fig = px.bar(f_counts, x="Count", y="folder_name", orientation="h", color_discrete_sequence=PALETTE)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with c4:
            if "doc_type_name" in f_docs.columns and not f_docs.empty:
                st.markdown("**📋 Document Type Breakdown**")
                dt_counts = f_docs["doc_type_name"].fillna("(No Type)").value_counts().reset_index()
                dt_counts.columns = ["doc_type_name", "Count"]
                type_count = len(dt_counts)
                st.caption(f"📑 **{type_count}** document types")
                fig = px.pie(dt_counts, names="doc_type_name", values="Count", hole=0.4, color_discrete_sequence=PALETTE)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Visibility
        if "visibility" in f_docs.columns and not f_docs.empty:
            st.markdown("---")
            st.markdown("**🔐 Access & Visibility Control**")
            vis_all_options = ["ALL", "DEPARTMENT", "SPECIFIC_USERS", "SPECIFIC_ROLES", "ROLE_DEPARTMENT"]
            v_counts = f_docs["visibility"].value_counts().reindex(vis_all_options, fill_value=0).reset_index()
            v_counts.columns = ["visibility", "Count"]
            public_docs = v_counts[v_counts["visibility"] == "ALL"]["Count"].sum()
            restricted_docs = v_counts[v_counts["visibility"] != "ALL"]["Count"].sum()
            if public_docs > restricted_docs:
                st.caption(f"🌐 **Open Access**: {public_docs} documents are publicly visible")
            else:
                st.caption(f"🔒 **Controlled Access**: {restricted_docs} documents have restricted visibility")
            fig = px.bar(v_counts, x="visibility", y="Count", color_discrete_sequence=PALETTE)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents found or failed to load documents.")


elif page == "📋 Requests Analytics":
    # Header
    year_display = selected_year if selected_year != "All Years" else "All Time"
    st.markdown(f"### 📋 Requests Analytics - {year_display}")
    st.markdown("*Visual insights into request workflow: completion rates, priority distribution, and workload analysis.*")
    
    reqs_df = reqs_df_full.copy()
    if not reqs_df.empty: 

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
        
        st.markdown("---")
        
        # Charts only - no tables or filters
        f_reqs = reqs_df.copy()
        
        # Row 1: Status & Priority
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**📊 Request Status Distribution**")
            status_counts = f_reqs["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(status_counts, names="Status", values="Count", hole=0.4, color_discrete_sequence=PALETTE)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with r2:
            if "priority" in f_reqs.columns:
                st.markdown("**⚡ Priority Distribution**")
                priority_counts = f_reqs["priority"].value_counts().reset_index()
                priority_counts.columns = ["Priority", "Count"]
                urgent_count = priority_counts[priority_counts["Priority"] == "urgent"]["Count"].sum() if "urgent" in priority_counts["Priority"].values else 0
                st.caption(f"🚨 **{urgent_count}** urgent requests")
                fig = px.bar(priority_counts, x="Priority", y="Count", color_discrete_sequence=PALETTE)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Role & Timeline
        st.markdown("---")
        r3, r4 = st.columns(2)
        
        with r3:
            if "assigned_to_role" in f_reqs.columns:
                st.markdown("**👥 Workload by Role**")
                role_counts = f_reqs["assigned_to_role"].value_counts().reset_index()
                role_counts.columns = ["Role", "Count"]
                busiest_role = role_counts.iloc[0]["Role"] if not role_counts.empty else "N/A"
                busiest_count = role_counts.iloc[0]["Count"] if not role_counts.empty else 0
                st.caption(f"Most assigned: **{busiest_role}** ({busiest_count} requests)")
                fig = px.bar(role_counts, x="Count", y="Role", orientation="h", color_discrete_sequence=PALETTE)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with r4:
            if "created_at" in f_reqs.columns and not f_reqs.empty:
                st.markdown("**📈 Requests Over Time**")
                ts_data = f_reqs.dropna(subset=["created_at"]).copy()
                ts_data["month"] = pd.to_datetime(ts_data["created_at"]).dt.to_period("M").dt.to_timestamp()
                ts = ts_data.groupby("month").size().reset_index(name="Requests")
                fig = px.line(ts, x="month", y="Requests", markers=True)
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data available")
    else:
        st.info("No document requests found or failed to load requests.")
 

