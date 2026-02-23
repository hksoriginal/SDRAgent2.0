import streamlit as st
import httpx
import asyncio
import pandas as pd

# ------------------------------
# CONFIGURATION
# ------------------------------
BACKEND_URL = "http://127.0.0.1:8585"

st.set_page_config(
    page_title="SDR Agent Chat",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Sales Development Representative (SDR) Chat")

# ------------------------------
# ASYNC HELPERS
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
# CHAT INTERFACE
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "😹"):
        st.markdown(msg["content"])
        if msg.get("data") is not None:
            df = msg["data"]
            if not df.empty:
                st.dataframe(df, use_container_width=True)

# Input area for new message
if prompt := st.chat_input("Ask me anything about your leads..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Placeholder for assistant reply
    with st.chat_message("assistant", avatar="😹"):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()

            try:
                # Call the /run_graph endpoint
                result = run_async_task(async_post(
                    "/run_graph", {"query": prompt}))
                final_state = result.get("final_state", {}) or {}

                # st.json(final_state)

                # Extract from final_state
                query_result_data = final_state.get("query_result", {}) or {}
                rows = query_result_data.get(
                    "query_result", {}).get("rows", []) or []

                # Prepare DataFrame if rows exist
                df = None
                if rows:
                    df = pd.DataFrame(rows)
                    # Prioritize important columns (Lead ID/Number first)
                    priority_cols = [
                        c for c in df.columns if "ID" in c or "Lead Number" in c
                    ]
                    df = df[priority_cols +
                            [c for c in df.columns if c not in priority_cols]]

                # Display assistant response
                ai_reply = f"{result.get('message', 'No message returned from agent.')}"
                message_placeholder.markdown(ai_reply)

                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"✅ Displayed {len(df)} records successfully.")

                # Save assistant message
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_reply, "data": df}
                )

            except httpx.HTTPStatusError as e:
                error_msg = f"❌ Backend error ({e.response.status_code}): {e.response.text}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

            except Exception as e:
                error_msg = f"❌ Pipeline execution failed: {e}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
