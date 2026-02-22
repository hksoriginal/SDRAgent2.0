import time
import uuid
import logging
from langgraph.graph import StateGraph, END
from Utils.intent_classifier import IntentClassifier
from Agents.DataQueryAgent.query_agent import DataQueryAgent
from Agents.EmailAgent.email_agent import EmailAgent

logger = logging.getLogger(__name__)


# --- Helper Function: Tracing Decorator --- #
def traced_node(node_name: str):
    """
    Decorator to add structured logging and execution timing to graph nodes.
    Each node call will be logged with trace_id for correlation across services.
    """
    def decorator(func):
        async def wrapper(state: dict):
            trace_id = state.get("trace_id") or str(uuid.uuid4())
            state["trace_id"] = trace_id
            start_time = time.perf_counter()
            logger.info(
                f"[TRACE_ID={trace_id}] ▶️ Entering node: {node_name} | State keys: {list(state.keys())}"
            )
            try:
                result = await func(state)
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"[TRACE_ID={trace_id}] ✅ Node completed: {node_name} | Duration: {elapsed:.2f}s"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.exception(
                    f"[TRACE_ID={trace_id}] ❌ Error in node: {node_name} | Duration: {elapsed:.2f}s | Error: {e}"
                )
                raise
        return wrapper
    return decorator


# --- Node wrapper functions --- #
@traced_node("intent_classifier")
async def intent_classifier_node(state: dict) -> dict:
    """Node wrapper for intent classification."""
    classifier = IntentClassifier()
    user_input = state.get("user_input", "")
    result = await classifier.classify_intent(user_input=user_input)
    state.update(
        {"intent": result["intent"],
            "confidence": result["confidence"], "intent_meta": result}
    )
    return state


@traced_node("data_query")
async def data_query_node(state: dict) -> dict:
    """Node wrapper for data querying."""
    agent = DataQueryAgent()
    user_input = state.get("user_input", "")
    result = await agent.query(user_input)
    state.update({"query_result": result})
    return state


@traced_node("send_email")
async def email_node(state: dict) -> dict:
    """Node wrapper for sending or drafting emails."""
    agent = EmailAgent()
    user_input = state.get("user_input", "")
    result = await agent.get_email_ids()
    state.update({"email_result": result})  
    return state


# --- Graph Builder --- #
async def build_langgraph_pipeline():
    """Builds the complete LangGraph flow with tracing logs."""
    trace_id = str(uuid.uuid4())
    logger.info(f"[TRACE_ID={trace_id}] ⚙️ Building LangGraph pipeline...")

    graph = StateGraph(dict)  # schema can be replaced with TypedDict

    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("data_query", data_query_node)
    graph.add_node("send_email", email_node)

    # Conditional flow routing
    graph.add_conditional_edges(
        "intent_classifier",
        lambda state: state.get("intent", "fallback"),
        {
            "query_data": "data_query",
            "send_email": "send_email",
            "fallback": END,
        },
    )

    graph.set_entry_point("intent_classifier")

    logger.info(
        f"[TRACE_ID={trace_id}] ✅ LangGraph pipeline built successfully.")
    return graph
