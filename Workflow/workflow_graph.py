import time
import logging
from langgraph.graph import StateGraph, END
from Utils.intent_classifier import IntentClassifier
from Utils.data_processor import DataProcessor
from Agents.DataQueryAgent.query_agent import DataQueryAgent
from Agents.EmailAgent.email_agent import EmailAgent
from Mixins.llm_client import OpenRouterMixin


logger = logging.getLogger(__name__)


async def intent_classifier_node(state: dict) -> dict:
    """Node wrapper for intent classification."""
    logger.info("[INTENT_CLASSIFIER_NODE] Starting intent classification...")
    classifier = IntentClassifier()
    user_input = state.get("user_input", "")
    result = await classifier.classify_intent(user_input=user_input)
    state.update(
        {"intent": result["intent"]}
    )
    return state


async def data_query_node(state: dict) -> dict:
    """Node wrapper for data querying."""
    logger.info("[DATA_QUERY_NODE] Starting data query...")
    agent = DataQueryAgent()
    user_input = state.get("user_input", "")
    result = await agent.query(user_input)
    print(result.keys())
    state.update(
        {"query_result": result, "message": result["summary"]})
    return state


async def email_node(state: dict) -> dict:
    """Node wrapper for sending or drafting emails."""
    logger.info("[EMAIL_NODE] Starting email processing...")
    agent = EmailAgent()
    result = await agent.get_email_ids()
    state.update(
        {"email_result": result, "message": f"{len(result)} Emails has been sent successfully."})
    return state


async def refresh_data_node(state: dict) -> dict:
    logger.info("[REFRESH_DATA_NODE] Starting data refresh...")
    data_processor = DataProcessor(
        df1_path="Datafiles/Leads.csv",
        df2_path="Datafiles/SampleData.csv"
    )

    filtered_df = data_processor.get_filter_data(threshold=0.45)
    data_processor.save_filtered_dataframe(
        filtered_dataframe=filtered_df,
        path="Datafiles/filtered_leads.csv"
    )
    state.update({"message": "Data refreshed successfully."})
    return state


async def fallback_node(state: dict) -> dict:
    logger.info("[FALLBACK_NODE] Executing fallback response...")
    llm = OpenRouterMixin()
    user_input = state.get("user_input", "")
    result = await llm.chat_completion(
        messages=[
            {"role": "user", "content": f"You are a Sales Development Agent 'Harshit' at 'Ema'. Fallback response for: {user_input}"}],
        model="meta-llama/llama-3.3-70b-instruct",
        temperature=0.5,
        max_tokens=50)
    state.update({"message": result})
    return state
