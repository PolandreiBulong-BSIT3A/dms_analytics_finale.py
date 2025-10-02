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
st.set_page_config(page_title="ISPSC Tagudin DMS Analytics", layout="wide")
st.title("ISPSC Tagudin DMS Analytics Dashboard")
st.caption("Users • Documents • Announcements")

# Refresh control
hdr1, hdr2 = st.columns([0.2, 1])
with hdr1:
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
dept_q = """
    SELECT d.department_id, d.name AS department_name, d.code
    FROM departments d
"""
users_q = """
    SELECT u.user_id, u.Username, u.firstname, u.lastname, u.user_email,
           u.role, u.status, u.is_verified, u.department_id, u.created_at, u.updated_at
    FROM dms_user u
"""
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
ann_q = """
    SELECT announcement_id, title, visible_to_all, status, publish_at, expire_at,
           created_by_name, created_at, updated_at
    FROM announcements
"""

data = fetch_many(engine_uri, {"departments": dept_q, "users": users_q, "docs": docs_q, "ann": ann_q})
departments = data["departments"]

# Global chart style
PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]
px.defaults.template = "plotly_dark"

# Minimal layout: tabs per section
users_tab, docs_tab, ann_tab = st.tabs(["Users", "Documents", "Announcements"])

# -----------------------------
# Users
# -----------------------------
with users_tab:
    st.caption("Users")
    users_df = data["users"].copy()
    if not users_df.empty:
        users_df = users_df.merge(departments, how="left", on="department_id")
        users_df.rename(columns={"department_name": "Department", "role": "Role", "status": "Status"}, inplace=True)
        users_df = to_dt(users_df, ["created_at", "updated_at"]) 

        # KPI cards
        u_k1, u_k2, u_k3 = st.columns(3)
        with u_k1:
            st.metric("Total Users", f"{len(users_df):,}")
        with u_k2:
            st.metric("Active Users", f"{(users_df['Status'] == 'active').sum():,}")
        with u_k3:
            st.metric("Verified Users", f"{(users_df['is_verified'] == 'yes').sum():,}")

        ucol_f, ucol_s = st.columns(2)
        with ucol_f:
            with st.expander("Filters", expanded=False):
                roles = sorted([r for r in users_df["Role"].dropna().unique()])
                role_sel = st.multiselect("Role", roles, default=roles, key="user_role")
                statuses = sorted([s for s in users_df["Status"].dropna().unique()])
                status_sel = st.multiselect("Status", statuses, default=statuses, key="user_status")
                depts = sorted([d for d in users_df["Department"].dropna().unique()])
                dept_sel = st.multiselect("Department", depts, default=depts, key="user_dept")
                # Date range
                if "created_at" in users_df.columns and not users_df["created_at"].isna().all():
                    min_dt = pd.to_datetime(users_df["created_at"].min()).date()
                    max_dt = pd.to_datetime(users_df["created_at"].max()).date()
                    start, end = st.date_input("Created between", value=(min_dt, max_dt), key="user_created_between")
                else:
                    start, end = None, None

        f_users = users_df.copy()
        if role_sel:
            f_users = f_users[f_users["Role"].isin(role_sel)]
        if status_sel:
            f_users = f_users[f_users["Status"].isin(status_sel)]
        if dept_sel:
            f_users = f_users[f_users["Department"].isin(dept_sel)]
        if start and end and "created_at" in f_users.columns:
            f_users = f_users[(f_users["created_at"].dt.date >= start) & (f_users["created_at"].dt.date <= end)]
        # Default sort and table
        f_users = apply_sort(f_users, "user_id", True)
        st.dataframe(f_users, use_container_width=True, hide_index=True)

        # CSV export for Users
        csv_u = f_users.to_csv(index=False).encode('utf-8')
        st.download_button("Download Users (CSV)", data=csv_u, file_name="users_filtered.csv", mime="text/csv")

        # Chart toggles
        with st.expander("Chart toggles", expanded=False):
            show_u_role = st.checkbox("Users by Role", value=True, key="u_role")
            u_role_type = st.selectbox("Users by Role chart type", ["Pie", "Bar"], index=0, key="u_role_type")
            show_u_dept = st.checkbox("Users by Department", value=True, key="u_dept")
            u_dept_type = st.selectbox("Users by Department chart type", ["Bar", "Pie"], index=0, key="u_dept_type")
            # Users over time
            show_u_time = st.checkbox("Users over Time (created_at)", value=True, key="u_time")
            u_gran_map = {"Day": "D", "Week": "W", "Month": "M"}
            u_gran = st.selectbox("Users time granularity", list(u_gran_map.keys()), index=0, key="u_time_gran")
            u_time_type = st.selectbox("Users time chart type", ["Line", "Bar"], index=0, key="u_time_type")

        c1, c2 = st.columns(2)
        with c1:
            if show_u_role:
                st.caption("Users by Role")
                role_counts = f_users["Role"].value_counts().reset_index()
                role_counts.columns = ["Role", "Count"]
                if u_role_type == "Pie":
                    fig = px.pie(role_counts, names="Role", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
                else:
                    fig = px.bar(role_counts, x="Role", y="Count", color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if show_u_dept:
                st.caption("Users by Department")
                dept_counts = f_users["Department"].value_counts().reset_index()
                dept_counts.columns = ["Department", "Count"]
                if u_dept_type == "Bar":
                    fig = px.bar(dept_counts, x="Department", y="Count", color_discrete_sequence=PALETTE)
                else:
                    fig = px.pie(dept_counts, names="Department", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, use_container_width=True)

        # Users over time
        if show_u_time and "created_at" in f_users.columns and not f_users.empty:
            ts_u = (
                f_users.dropna(subset=["created_at"]) 
                       .groupby(f_users["created_at"].dt.to_period(u_gran_map[u_gran])).size()
                       .reset_index(name="Count")
            )
            ts_u["date"] = ts_u["created_at"].dt.to_timestamp()
            st.caption("Users over Time (created_at)")
            if u_time_type == "Line":
                st.plotly_chart(px.line(ts_u, x="date", y="Count"), use_container_width=True)
            else:
                st.plotly_chart(px.bar(ts_u, x="date", y="Count", color_discrete_sequence=PALETTE), use_container_width=True)
    else:
        st.info("No users found or failed to load users.")


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
                vis_vals = sorted([v for v in docs_df["visibility"].dropna().unique()])
                vis_sel = st.multiselect("Visibility", vis_vals, default=vis_vals, key="doc_visibility")
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
            v_counts = f_docs["visibility"].fillna("(None)").value_counts().reset_index()
            v_counts.columns = ["visibility", "Count"]
            if d_visibility_type == "Pie":
                fig = px.pie(v_counts, names="visibility", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
            else:
                fig = px.bar(v_counts, x="visibility", y="Count", color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents found or failed to load documents.")


# -----------------------------
# Announcements
# -----------------------------
with ann_tab:
    st.caption("Announcements")
    ann_df = data["ann"].copy()
    if not ann_df.empty:
        ann_df = to_dt(ann_df, ["publish_at", "expire_at", "created_at", "updated_at"]) 
        if "visible_to_all" in ann_df.columns:
            ann_df["visible_to_all"] = ann_df["visible_to_all"].map({1: True, 0: False}).fillna(ann_df["visible_to_all"])

        # KPI cards
        a_k1, a_k2, a_k3 = st.columns(3)
        with a_k1:
            st.metric("Total Announcements", f"{len(ann_df):,}")
        with a_k2:
            st.metric("Published (status)", f"{(ann_df['status'] == 'published').sum():,}")
        with a_k3:
            st.metric("Visible to All", f"{(ann_df.get('visible_to_all') == True).sum():,}")

        acol_f, acol_s = st.columns(2)
        with acol_f:
            with st.expander("Filters", expanded=False):
                creators = sorted([c for c in ann_df["created_by_name"].dropna().unique()]) if "created_by_name" in ann_df.columns else []
                creator_sel = st.multiselect("Created by", creators, default=creators, key="ann_creator")
                visible_sel = st.multiselect("Visible to All", [True, False], default=[True, False], key="ann_visible")
                if "publish_at" in ann_df.columns and not ann_df["publish_at"].isna().all():
                    amin = pd.to_datetime(ann_df["publish_at"].min()).date()
                    amax = pd.to_datetime(ann_df["publish_at"].max()).date()
                    astart, aend = st.date_input("Publish date between", value=(amin, amax), key="ann_date_between")
                else:
                    astart, aend = None, None

        f_ann = ann_df.copy()
        if creator_sel and "created_by_name" in f_ann.columns:
            f_ann = f_ann[f_ann["created_by_name"].isin(creator_sel)]
        if visible_sel and "visible_to_all" in f_ann.columns:
            f_ann = f_ann[f_ann["visible_to_all"].isin(visible_sel)]
        if astart and aend and "publish_at" in f_ann.columns:
            f_ann = f_ann[(f_ann["publish_at"].dt.date >= astart) & (f_ann["publish_at"].dt.date <= aend)]

        # Default sort (no UI)
        sort_col_a = "announcement_id"
        asc_a = True
        f_ann = apply_sort(f_ann, sort_col_a, asc_a)
        st.dataframe(f_ann, use_container_width=True, hide_index=True)

        # CSV export
        csv_a = f_ann.to_csv(index=False).encode('utf-8')
        st.download_button("Download Announcements (CSV)", data=csv_a, file_name="announcements_filtered.csv", mime="text/csv")

        with st.expander("Chart toggles", expanded=False):
            show_a_creator = st.checkbox("Announcements by Creator", value=True, key="a_creator")
            a_creator_type = st.selectbox("Announcements by Creator chart type", ["Pie", "Bar"], index=0, key="a_creator_type")
            show_a_time = st.checkbox("Announcements Published Over Time", value=True, key="a_time")
            a_time_type = st.selectbox("Announcements time chart type", ["Line", "Bar"], index=0, key="a_time_type")
            granularity_map_a = {"Day": "D", "Week": "W", "Month": "M"}
            granularity_a = st.selectbox("Time granularity", list(granularity_map_a.keys()), index=0, key="ann_granularity")

        c5, c6 = st.columns(2)
        with c5:
            if show_a_creator:
                st.caption("Announcements by Creator")
                if "created_by_name" in f_ann.columns and not f_ann.empty:
                    a_counts = f_ann["created_by_name"].value_counts().reset_index()
                    a_counts.columns = ["created_by_name", "Count"]
                    if a_creator_type == "Pie":
                        fig = px.pie(a_counts, names="created_by_name", values="Count", hole=0.3, color_discrete_sequence=PALETTE)
                    else:
                        fig = px.bar(a_counts, x="created_by_name", y="Count", color_discrete_sequence=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No creator data to plot.")
        with c6:
            if show_a_time:
                st.caption("Announcements Published Over Time")
                if "publish_at" in f_ann.columns and not f_ann.empty:
                    period_a = granularity_map_a[granularity_a]
                    ts = (
                        f_ann.dropna(subset=["publish_at"]) 
                             .groupby(f_ann["publish_at"].dt.to_period(period_a)).size()
                             .reset_index(name="Count")
                    )
                    ts["date"] = ts["publish_at"].dt.to_timestamp()
                    if a_time_type == "Line":
                        fig = px.line(ts, x="date", y="Count")
                    else:
                        fig = px.bar(ts, x="date", y="Count", color_discrete_sequence=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No publish_at values to plot.")
    else:
        st.info("No announcements found or failed to load announcements.")



