import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()

            # Drop irrelevant columns
            drop_cols = ["Customer ID", "Name", "Type of Employment", "Property ID"]
            df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

            # Replace -999 with 0
            df.replace(-999, 0, inplace=True)

            # Drop rare professions
            if "Profession" in df.columns:
                rare_professions = ["Unemployed", "Businessman", "Student", "Maternity leave"]
                df = df[~df["Profession"].isin(rare_professions)]

            # Fill missing values
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

            # Drop any remaining missing values
            df.dropna(inplace=True)

            # Convert Property Age from days to years
            if "Property Age" in df.columns:
                df["Property Age"] = (df["Property Age"] / 365)

            # Remove outliers using IQR
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]

            # Update num_cols after outlier removal
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns
            
            # Handle skewness using cube root transformation
            
            skewed = df[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
            skewed_cols = skewed[skewed.abs() > 1].index
            for col in skewed_cols:
                 if col != "Loan Sanction Amount (USD)":
                      df[col] = np.cbrt(df[col])
                      
            # One-hot encode categorical columns
            cat_cols = df.select_dtypes(include=["object"]).columns
            df = pd.get_dummies(df, columns=cat_cols)
            
            # Min-Max Scaling (excluding target)
            scale_cols = df.select_dtypes(include=["float64", "int64"]).columns
            scale_cols = [col for col in scale_cols if col != "Loan Sanction Amount (USD)"]
            scaler = MinMaxScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            
            logging.info("Data preprocessing complete.")
            return df

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["Loan Sanction Amount (USD)"], axis=1)
            y = data["Loan Sanction Amount (USD)"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise

class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e