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
        ["📈 Dashboard", "📄 Documents Analytics"],
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
        st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.9rem;'>Total Documents<br/><span style='font-size: 1.5rem; font-weight: 600; color: #111827;'>{len(docs_df_full):,}</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    docs_df = docs_df_full.copy()
    
    # Top row - Key metrics (6 cards)
    st.markdown("#### 📊 Key Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    total_docs = len(docs_df)
    active_docs = (docs_df['status'] == 'active').sum() if not docs_df.empty else 0
    inactive_docs = total_docs - active_docs
    deleted_docs = (docs_df.get('deleted', 0) == 1).sum() if not docs_df.empty else 0
    folder_count = docs_df["folder_name"].nunique() if not docs_df.empty and "folder_name" in docs_df.columns else 0
    doc_type_count = docs_df["doc_type_name"].nunique() if not docs_df.empty and "doc_type_name" in docs_df.columns else 0
    
    with m1:
        st.metric("Total Documents", f"{total_docs:,}", help="All documents in the system")
    with m2:
        st.metric("Active Docs", f"{active_docs:,}", help="Currently active documents")
    with m3:
        st.metric("Inactive Docs", f"{inactive_docs:,}", help="Inactive documents")
    with m4:
        st.metric("Deleted Docs", f"{deleted_docs:,}", help="Deleted documents")
    with m5:
        st.metric("Folders", f"{folder_count:,}", help="Number of folders")
    with m6:
        st.metric("Doc Types", f"{doc_type_count:,}", help="Number of document types")
    
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
        st.markdown("**📊 Document Status Distribution**")
        if not docs_df.empty and "status" in docs_df.columns:
            status_counts = docs_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_donut = px.pie(status_counts, names="Status", values="Count", hole=0.5)
            fig_donut.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=True)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No status data available")
    
    st.markdown("---")
    
    # Third row - Department Assignment Chart
    st.markdown("---")
    st.markdown("**🏢 Document Assignment by Department**")
    
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
        st.caption(f"📌 **{all_count}** documents accessible to all departments")
        
        fig_dept = px.bar(dept_counts, x="Department", y="Count", color_discrete_sequence=PALETTE)
        fig_dept.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_dept, use_container_width=True)
    else:
        st.info("No department data available")

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
            st.caption(f"📌 **{all_count}** documents accessible to all departments")
            
            fig = px.bar(dept_counts, x="Department", y="Count", color_discrete_sequence=PALETTE)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents found or failed to load documents.")
 

