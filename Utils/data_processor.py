import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A class to handle merging, filtering, and cleaning of two CSV datasets based on lead information.
    """

    def __init__(self, df1_path: str, df2_path: str):
        """
        Initializes the DataProcessor by loading two CSV files into DataFrames.
        """
        try:
            self.df1 = pd.read_csv(df1_path)
            logger.info(f"Successfully loaded first CSV from {df1_path}")
            self.df2 = pd.read_csv(df2_path, encoding="ISO-8859-1")
            logger.info(f"Successfully loaded second CSV from {df2_path}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    def _merge_dataframes(self, merge_on_column: str) -> pd.DataFrame:
        """
        Merges two DataFrames on a specified column, avoiding duplication on common columns.
        """
        try:
            common_columns = set(self.df1.columns).intersection(set(self.df2.columns))
            common_columns.remove(merge_on_column)
            merged_df = pd.merge(
                self.df1,
                self.df2.drop(columns=list(common_columns)),
                on=merge_on_column,
                how="outer"
            )
            logger.info(f"DataFrames merged successfully on '{merge_on_column}' column.")
            return merged_df
        except KeyError as e:
            logger.error(f"Column not found for merging: {e}")
            raise
        except Exception as e:
            logger.error(f"Error while merging DataFrames: {e}")
            raise

    def _drop_columns_with_missing_threshold(self, df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
        """
        Drops columns from the DataFrame if the percentage of missing/empty/null values
        exceeds a given threshold.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
            threshold (float): Threshold (0–1). Columns with missing ratio > threshold are dropped.

        Returns:
            pd.DataFrame: Cleaned DataFrame with columns removed.
        """
        try:
            # Count NaN, empty string, and None
            missing_ratio = df.isnull().mean()
            empty_ratio = (df.astype(str).apply(lambda x: x.str.strip() == "").mean())
            total_missing_ratio = (missing_ratio + empty_ratio) / 2  # Weighted average

            cols_to_drop = total_missing_ratio[total_missing_ratio > threshold].index
            df = df.drop(columns=cols_to_drop)

            logger.info(f"Dropped {len(cols_to_drop)} columns exceeding {threshold*100:.0f}% missing threshold.")
            if len(cols_to_drop) > 0:
                logger.info(f"Columns dropped: {list(cols_to_drop)}")

            return df
        except Exception as e:
            logger.error(f"Error while dropping columns: {e}")
            raise

    def get_filter_data(self, threshold: float = 0.6) -> pd.DataFrame:
        """
        Merges, filters, and cleans the DataFrame to include only relevant leads and remove
        columns with excessive missing values.

        Args:
            threshold (float): Missing-value threshold for dropping columns (default: 0.6).

        Returns:
            pd.DataFrame: Final cleaned DataFrame.
        """
        try:
            merged_df = self._merge_dataframes(merge_on_column="Lead Number")

            # Filter: only 'Landing Page Submission' and non-empty Company field
            filtered_df = merged_df[merged_df["Lead Origin"] == "Landing Page Submission"]
            filtered_df = filtered_df.dropna(subset=["Company"])

            # Drop columns with high missing ratio
            cleaned_df = self._drop_columns_with_missing_threshold(filtered_df, threshold)

            logger.info(f"Final processed DataFrame shape: {cleaned_df.shape}")
            return cleaned_df
        except KeyError as e:
            logger.error(f"Required column missing during filtering: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during data filtering: {e}")
            raise

    def save_filtered_dataframe(self, filtered_dataframe: pd.DataFrame, path: str, index: bool = False) -> str:
        """
        Saves a DataFrame to a CSV file.
        """
        try:
            filtered_dataframe.to_csv(path, index=index)
            logger.info(f"DataFrame successfully saved to {path}")
            return f"DataFrame successfully saved to {path}"
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            raise