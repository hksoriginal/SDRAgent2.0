import re
import pandas as pd
from Mixins.text_processor import TextProcessor
import logging

logger = logging.getLogger(__name__)


class EmailExtractor(TextProcessor):
    """
    Extracts valid email leads from a DataFrame.
    """

    def __init__(self):
        super().__init__()
        # Precompile regex once for efficiency
        self.email_regex = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')

    def get_lead_email_df(
        self,
        df: pd.DataFrame,
        email_col: str = "Company",
        required_cols: list = None,
        return_format: str = "json"  # "df" or "json"
    ):
        """
        Returns filtered email leads as either a DataFrame or JSON.

        Args:
            df (pd.DataFrame): Input DataFrame.
            email_col (str): Name of the column containing email addresses.
            required_cols (list): Additional non-null columns.
            return_format (str): Either 'df' or 'json'.

        Returns:
            pd.DataFrame | str: Filtered email DataFrame or JSON string.
        """
        columns_to_check = ["Notes", email_col]
        if required_cols:
            columns_to_check.extend(
                [col for col in required_cols if col not in columns_to_check]
            )

        try:
            df_filtered = df.copy()
            df_filtered.columns = [col.strip() for col in df_filtered.columns]

            existing_cols = [
                col for col in columns_to_check if col in df_filtered.columns
            ]
            missing_cols = [
                col for col in columns_to_check if col not in df_filtered.columns
            ]
            if missing_cols:
                logger.warning(
                    f"Missing columns in DataFrame: {missing_cols}. Ignoring them."
                )

            df_filtered = df_filtered.dropna(subset=existing_cols)

            if email_col in df_filtered.columns:
                df_filtered = df_filtered[
                    df_filtered[email_col].apply(
                        lambda x: bool(self.email_regex.match(str(x)))
                    )
                ]
            else:
                logger.warning(
                    f"Email column '{email_col}' not found; skipping email validation."
                )

            final_cols = [
                col for col in columns_to_check if col in df_filtered.columns
            ]
            result_df = df_filtered[final_cols]

            if return_format == "json":
                return result_df.to_dict(orient="records")
            return result_df

        except Exception as e:
            logger.exception(f"Error in get_lead_email_df: {e}")
            empty_df = pd.DataFrame(columns=columns_to_check)
            return empty_df.to_json(orient="records", indent=2) if return_format == "json" else empty_df
