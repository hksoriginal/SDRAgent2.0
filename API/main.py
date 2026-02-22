from typing import TypedDict, Optional, Any, Dict
from fastapi import FastAPI, Request, HTTPException
import logging
import asyncio

# Import your utilities and agents
from Utils.intent_classifier import IntentClassifier
from Agents.DataQueryAgent.query_agent import DataQueryAgent
from Utils.data_processor import DataProcessor
from Config.logging_config import configure_logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from Workflow.workflow_graph import (
    intent_classifier_node,
    data_query_node,
    email_node
)

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI()
intent_classifier = IntentClassifier()

# ---------- Existing Endpoints ---------- #


@app.post("/query")
async def query(request: Request):
    """Direct query execution (bypasses LangGraph)."""
    data = await request.json()
    user_query = data.get("query", "")
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    intent = await intent_classifier.classify_intent(user_input=user_query)
    agent = DataQueryAgent()
    res = await agent.query(user_query)

    return {"result": str(res), "intent": str(intent)}


@app.post("/refresh_data")
async def refresh_data(request: Request):
    """Refresh and filter lead data."""
    data = await request.json()
    refresh_flag = bool(data.get("refresh", False))
    if not refresh_flag:
        logger.warning("Invalid refresh flag received in request.")
        raise HTTPException(status_code=400, detail="Invalid refresh flag")

    try:
        data_processor = DataProcessor(
            df1_path="Datafiles/Leads.csv",
            df2_path="Datafiles/SampleData.csv"
        )

        filtered_df = data_processor.get_filter_data(threshold=0.45)
        data_processor.save_filtered_dataframe(
            filtered_dataframe=filtered_df,
            path="Datafiles/filtered_leads.csv"
        )

        logger.info("Data refreshed successfully.")
        return {"message": "Data refreshed successfully"}

    except Exception as e:
        logger.exception("Error while refreshing data: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh data: {str(e)}")


# ---------- New LangGraph Endpoint ---------- #


class GraphState(TypedDict, total=False):
    """Defines the expected state structure for the LangGraph."""
    user_input: str
    intent: str
    confidence: float
    query_result: Optional[Dict[str, Any]]
    email_result: Optional[Dict[str, Any]]
    summary: Optional[str]
    intent_meta: Optional[Dict[str, Any]]


@app.post("/run_graph")
async def run_graph(request: Request):
    """
    Run the complete LangGraph pipeline.
    Classifies intent, routes to correct node, and executes.
    """
    try:
        data = await request.json()
        user_query = data.get("query", "")
        if not user_query:
            raise HTTPException(
                status_code=400, detail="Missing 'query' field")

        logger.info(f"🧠 Running LangGraph pipeline for: '{user_query}'")

        # --- Build Graph Dynamically --- #
        graph_builder = StateGraph(GraphState)
        graph_builder.add_node("intent_classifier", intent_classifier_node)
        graph_builder.add_node("data_query", data_query_node)
        graph_builder.add_node("send_email", email_node)

        graph_builder.add_conditional_edges(
            "intent_classifier",
            lambda state: state.get("intent", "fallback"),
            {
                "query_data": "data_query",
                "send_email": "send_email",
                "summarize": END,
                "greeting": END,
                "fallback": END,
            },
        )

        graph_builder.set_entry_point("intent_classifier")
        graph = graph_builder.compile()

        # --- Execute Graph --- #
        state = {"user_input": user_query}
        result = await graph.ainvoke(state) 

        logger.info("✅ LangGraph pipeline completed successfully.")
        return {
            "message": "Graph executed successfully",
            "final_state": result,
        }

    except Exception as e:
        logger.exception(f"Error while running LangGraph pipeline: {e}")
        raise HTTPException(
            status_code=500, detail=f"Graph execution failed: {str(e)}")
