from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, avg
from pyspark.sql.types import IntegerType, StringType, FloatType
import pandas as pd

def initialize_spark():
    """
    Initialize a Spark session.
    """
    spark = SparkSession.builder \
        .appName("IDS Data Preprocessing") \
        .getOrCreate()
    return spark

def load_data(spark, train_path, test_path):
    """
    Load the training and testing datasets from parquet files.
    """
    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)
    
    return train_df, test_df

def inspect_data(df):
    """
    Inspect the DataFrame: show schema, basic statistics, and first few rows.
    """
    print("Schema:")
    df.printSchema()
    
    print("Basic Statistics:")
    df.describe().show()
    
    print("First 5 rows:")
    df.show(5)

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    """
    print("Handling missing values...")

    # Count missing values for each column
    missing_count = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    print("Missing values before handling:")
    missing_count.show()

    # Fill missing values: Strategy can be adjusted based on the domain knowledge
    # For simplicity, we can fill numeric columns with the mean and categorical with mode.
    for column in df.columns:
        if df.schema[column].dataType in [FloatType(), IntegerType()]:
            mean_value = df.select(avg(col(column))).first()[0]
            df = df.na.fill({column: mean_value})
        else:
            mode_value = df.groupBy(column).count().orderBy("count", ascending=False).first()[0]
            df = df.na.fill({column: mode_value})

    # Count missing values again
    missing_count_after = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    print("Missing values after handling:")
    missing_count_after.show()

    return df

def feature_extraction(df):
    """
    Extract features from the DataFrame as needed.
    Here we can add any domain-specific features that might help the LSTM model.
    """
    # Example: Convert categorical features to numeric using one-hot encoding
    categorical_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
    for col_name in categorical_cols:
        df = df.withColumn(col_name, when(col(col_name) == "normal", 1).otherwise(0))

    return df

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns that may not contribute to the model.
    """
    # Identify columns to drop; this is domain-specific and should be adapted as needed.
    columns_to_drop = ['id', 'ts', 'label']  # Example columns
    df = df.drop(*columns_to_drop)

    return df

def save_processed_data(df, output_path):
    """
    Save the processed DataFrame to a parquet file for future use.
    """
    df.write.parquet(output_path)

def main():
    train_path = "/home/kali/Bg01/Dataset/UNSW-NB15/UNSW_NB15_training-set.parquet"
    test_path = "/home/kali/Bg01/Dataset/UNSW-NB15/UNSW_NB15_testing-set.parquet"
    output_train_path = "/home/kali/Bg01/Dataset/UNSW-NB15/processed_train.parquet"
    output_test_path = "/home/kali/Bg01/Dataset/UNSW-NB15/processed_test.parquet"

    # Initialize Spark
    spark = initialize_spark()

    # Load data
    train_df, test_df = load_data(spark, train_path, test_path)

    # Inspect the training and testing data
    print("Inspecting Training Data:")
    inspect_data(train_df)

    print("Inspecting Testing Data:")
    inspect_data(test_df)

    # Handle missing values
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    # Feature extraction
    train_df = feature_extraction(train_df)
    test_df = feature_extraction(test_df)

    # Drop unnecessary columns
    train_df = drop_unnecessary_columns(train_df)
    test_df = drop_unnecessary_columns(test_df)

    # Save the processed data
    save_processed_data(train_df, output_train_path)
    save_processed_data(test_df, output_test_path)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
