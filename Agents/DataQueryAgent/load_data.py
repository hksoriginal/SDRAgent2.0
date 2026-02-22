import os
import logging
import traceback
import json
import pandas as pd
from typing import Any, Dict

# Try to import duckdb first (faster and better)
try:
    import duckdb
    _USE_DUCKDB = True
except ImportError:
    _USE_DUCKDB = False
    try:
        from pandasql import sqldf
    except ImportError as e:
        raise ImportError(
            "Neither 'duckdb' nor 'pandasql' is installed. Please install one:\n"
            "  pip install duckdb   # preferred\n"
            "  or\n"
            "  pip install pandasql"
        )

# Configure logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles CSV loading and SQL querying for DataFrames."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            data = pd.read_csv(file_path)
            logger.info("Data loaded successfully from %s. Shape: %s",
                        file_path, data.shape)
            return data

        except FileNotFoundError as e:
            logger.error(str(e))
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty: %s", file_path)
        except Exception as e:
            logger.error("Unexpected error while loading data: %s", str(e))
            logger.debug(traceback.format_exc())
        return None

    def query_data(self, sql_query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a SQL query on a pandas DataFrame using DuckDB (default) or pandasql.

        Args:
            sql_query (str): SQL query string. Use 'df' as the table name.
            data (pd.DataFrame): Input DataFrame.

        Returns:
            Dict[str, Any]: Query result as a dictionary with non-None rows only.
        """
        if data is None or data.empty:
            logger.warning("No data provided or DataFrame is empty.")
            return {"status": "error", "message": "No data available", "rows": []}

        if not sql_query or not sql_query.strip():
            logger.warning("Empty SQL query provided.")
            return {"status": "error", "message": "Empty SQL query", "rows": []}

        try:
            logger.info("Executing SQL query: %s", sql_query)

            if _USE_DUCKDB:
                duckdb.register("df", data)
                result = duckdb.query(sql_query).to_df()
            else:
                env = {"df": data}
                result = sqldf(sql_query, env)

            logger.info(
                "SQL query executed successfully — %d rows returned.", len(result))

            # ✅ Drop rows where all values are None
            result = result.dropna(how="all")

            # ✅ Convert to JSON (records = list of dicts)
            result_json = result.to_json(orient="records", date_format="iso")
            rows = json.loads(result_json)

            # ✅ Optionally remove keys with None values within each row
            cleaned_rows = [
                {k: v for k, v in row.items() if v is not None} for row in rows
            ]

            output = {
                "status": "success",
                "rows_returned": len(cleaned_rows),
                "rows": cleaned_rows
            }

        except Exception as e:
            logger.error("Error executing SQL query: %s", e)
            logger.debug(traceback.format_exc())

            output = {
                "status": "error",
                "message": str(e),
                "rows": []
            }

        return output
