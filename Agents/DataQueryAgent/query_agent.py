import asyncio
import logging
import traceback
import json
from typing import Any, Dict, List
from Agents.DataQueryAgent.load_data import DataLoader
from Mixins.llm_client import OpenRouterMixin
from Mixins.text_processor import TextProcessor

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logger = logging.getLogger(__name__)


SQL_GENERATION_PROMPT = """
You are an expert data analyst specializing in SQL generation for pandas DataFrames.

Task:
Generate an **executable SQL query** for a DataFrame named `df` based strictly on the user's natural language request.

Schema:
Columns = {columns}

Guidelines:
- use LIMIT 5 to restrict results for large datasets or not specified.
- Use only the provided columns.
- Assume column names are case-sensitive.
- If a column name contains spaces or special characters, enclose it in double quotes (e.g., "Lead Source").
- For partial or fuzzy text matching (e.g., contains, includes, similar to), use SQL LIKE syntax with wildcards.
  Example: WHERE "Company" LIKE '%University%'
- Always use SQLite/DuckDB-compatible SQL syntax.
- Do NOT use backticks (`) or single quotes (') for column names.
- Return **only** the SQL query string — no explanations, markdown, or formatting.

Context:
"Company" → organization name (e.g., "Pune University")
"Notes" → additional text about the lead
Other columns → may include country, specialization, state, age, etc.

User Request:
{user_query}

Output:
Return only the SQL query string.
"""

RESULT_SUMMARIZATION_PROMPT = """
You are a senior data analyst skilled at interpreting SQL query results and explaining insights clearly in natural language.

Task:
Analyze the provided SQL query result and write a concise, human-readable summary of the findings.

## Input
- SQL Query: {sql_query}
- Query Result (in JSON format): {query_result_json}

## Guidelines
- Use simple, professional, and factual language.
- If the result is a count or aggregation, explain what it represents (e.g., "There are 120 leads from Google").
- If multiple rows or groups are present, summarize key patterns (e.g., "Most leads come from universities, followed by startups").
- Do not invent or assume data not shown in the result.
- Avoid SQL jargon or technical terms like “rows”, “columns”, or “query output”.
- If the result is empty, respond: "No matching records were found."
- Keep responses under 80 words for quick readability.

## Output
Return only the natural-language summary.
"""


class DataQueryAgent(DataLoader, OpenRouterMixin, TextProcessor):
    """Agent to convert natural language queries into SQL queries."""

    def __init__(self):
        """Initialize DataQueryAgent with dataset and LLM client."""
        DataLoader.__init__(self)
        OpenRouterMixin.__init__(self)

        try:
            logger.info("Loading data from CSV...")
            self.data = self.load_data(
                file_path="Datafiles/filtered_leads.csv")
            logger.info("Data loaded successfully. Shape: %s", self.data.shape)
        except Exception as e:
            logger.error("Failed to load data: %s", str(e))
            logger.debug(traceback.format_exc())
            raise

    def get_column_context(self) -> str:
        """Prepare a concise context of available columns and unique values."""
        try:
            context: Dict[str, Any] = {}
            candidate_cols = [
                "Specialization", "Lead Source", "Country",
                "Lead Quality", "State", "Age", "City"
            ]

            for col in candidate_cols:
                if col in self.data.columns:
                    unique_vals = self.data[col].dropna().unique().tolist()
                    if "Select" in unique_vals:
                        unique_vals.remove("Select")
                    # Limit to top 10 for brevity
                    context[col] = unique_vals[:10]

            base_context = {
                "Company": "Organization name (e.g., 'Pune University')",
                "Notes": "Additional remarks about the lead",
            }

            full_context = {**base_context, **context}
            return json.dumps(full_context, indent=2)

        except Exception as e:
            logger.error("Error generating column context: %s", str(e))
            logger.debug(traceback.format_exc())
            raise

    async def generate_sql(self, user_query: str) -> str:
        """Generate SQL query using Llama 3.3 70B Instruct model via OpenRouter."""
        try:
            if not user_query.strip():
                raise ValueError("Empty user query provided.")

            logger.info(
                "Generating SQL query for user request: %s", user_query)
            column_context = self.get_column_context()
            prompt = SQL_GENERATION_PROMPT.format(
                columns=column_context, user_query=user_query)

            response = await self.chat_completion(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system",
                        "content": "You are a precise SQL generation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            sql_query = response.strip().strip("```sql").strip("```").strip()
            logger.info("Generated SQL: %s", sql_query)
            return sql_query

        except Exception as e:
            logger.error("Error generating SQL query: %s", str(e))
            logger.debug(traceback.format_exc())
            raise

    async def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute the generated SQL query on the DataFrame."""
        try:
            if not sql_query.strip():
                raise ValueError("Empty SQL query provided.")

            logger.info("Executing SQL query: %s", sql_query)
            result = self.query_data(sql_query, data=self.data)
            logger.info(
                "Query executed successfully.")
            return result

        except Exception as e:
            logger.error("Error executing SQL query: %s", str(e))
            logger.debug(traceback.format_exc())
            raise

    async def query(self, user_query: str) -> Dict[str, Any]:
        sql_query = await self.generate_sql(user_query)
        res = await self.execute_sql(sql_query)

        result_prompt = RESULT_SUMMARIZATION_PROMPT.format(
            sql_query=sql_query, query_result_json=self.get_toon(
                json=res.get("rows", []))
        )
        summary = await self.chat_completion(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "user", "content": result_prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        return {
            "sql_query": sql_query,
            "query_result": res,
            "summary": summary.strip()
        }
