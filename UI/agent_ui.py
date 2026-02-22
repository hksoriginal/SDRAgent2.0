import streamlit as st
import httpx
import asyncio
import pandas as pd
from datetime import datetime

# ------------------------------
# CONFIGURATION
# ------------------------------
BACKEND_URL = "http://127.0.0.1:8585"

st.set_page_config(
    page_title="AI Workflow Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Workflow Dashboard")
st.caption("A production-grade control panel for FastAPI + LangGraph backend")

# ------------------------------
# ASYNC REQUEST HANDLER
# ------------------------------


async def async_post(endpoint: str, payload: dict):
    """Asynchronous POST request to FastAPI backend."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{BACKEND_URL}{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()


def run_async_task(coro):
    """Helper to run async tasks inside Streamlit."""
    return asyncio.run(coro)


# ------------------------------
# SIDEBAR CONFIGURATION
# ------------------------------
st.sidebar.header("⚙️ Configuration")
BACKEND_URL = st.sidebar.text_input("Backend API URL", BACKEND_URL)
st.sidebar.markdown("---")
st.sidebar.info("Ensure FastAPI is running before using this dashboard.")


# ------------------------------
# TAB LAYOUT
# ------------------------------
tab1, tab2, tab3 = st.tabs(
    ["💬 Query Execution", "🔁 Data Refresh", "🧩 LangGraph Pipeline"])

# ------------------------------
# TAB 1 - DIRECT QUERY
# ------------------------------
with tab1:
    st.subheader("💬 Direct Query Execution")

    query_input = st.text_area(
        "Enter your query", placeholder="e.g., Get the top 3 leads working in universities")
    submit_query = st.button("Run Query", type="primary")

    if submit_query and query_input.strip():
        with st.spinner("Processing your query..."):
            try:
                result = run_async_task(async_post(
                    "/query", {"query": query_input}))
                st.success("✅ Query executed successfully!")
                st.json(result)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ------------------------------
# TAB 2 - DATA REFRESH
# ------------------------------
with tab2:
    st.subheader("🔁 Refresh and Filter Lead Data")

    st.markdown("Click below to refresh lead data using threshold filtering.")
    refresh_button = st.button("🔄 Refresh Data", type="primary")

    if refresh_button:
        with st.spinner("Refreshing data..."):
            try:
                result = run_async_task(async_post(
                    "/refresh_data", {"refresh": True}))
                st.success("✅ Data refreshed successfully!")
                st.json(result)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ------------------------------
# TAB 3 - LANGGRAPH PIPELINE
# ------------------------------
with tab3:
    st.subheader("🧩 Run LangGraph Workflow")

    graph_query = st.text_area(
        "Enter your query for the LangGraph pipeline",
        placeholder="e.g., Email the leads who work at Mumbai University"
    )
    run_graph_button = st.button("🚀 Run LangGraph Pipeline", type="primary")

    if run_graph_button and graph_query.strip():
        with st.spinner("Running LangGraph pipeline..."):
            try:
                result = run_async_task(async_post(
                    "/run_graph", {"query": graph_query}))
                st.success("✅ LangGraph pipeline executed successfully!")

                # --- Show full JSON response (optional) ---
                with st.expander("🧾 Full Response", expanded=False):
                    st.json(result)

                # --- Extract final state safely ---
                final_state = result.get("final_state", {}) or {}
                query_result_block = (
                    final_state.get("query_result", {}) or {}
                ).get("query_result", {}) or {}

                summary = final_state.get("query_result", {}).get(
                    "summary", "No summary available.")
                rows = query_result_block.get("rows", [])

                # --- Summary Section ---
                if summary:
                    st.markdown("### 🧠 Summary")
                    st.markdown(summary)
                else:
                    st.markdown("### 🧠 Summary")
                    st.warning("No summary found in the final state.")

                # --- Query Results Section ---
                if rows and isinstance(rows, list):
                    st.markdown("### 📊 Query Results")

                    try:
                        # Convert list[dict] → DataFrame
                        df = pd.DataFrame(rows)

                        # Optional: reorder key columns (ID, Lead Number, etc.)
                        priority_cols = [
                            c for c in df.columns if "ID" in c or "Lead Number" in c]
                        df = df[priority_cols +
                                [c for c in df.columns if c not in priority_cols]]

                        st.dataframe(df, use_container_width=True, height=500)
                        st.success(
                            f"✅ Displayed {len(df)} records successfully!")
                    except Exception as e:
                        st.error(f"⚠️ Error displaying query results: {e}")
                        st.json(rows[:2])  # fallback preview
                else:
                    st.info("No query result rows found in the LangGraph output.")

            except Exception as e:
                st.error(f"❌ Pipeline execution failed: {e}")
