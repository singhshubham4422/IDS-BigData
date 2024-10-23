from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, from_json
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from keras.models import load_model
import time

# Load the pre-trained model
model = load_model('/home/kali/Bg01/saved_model/enhanced_lstm_model.keras')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IDS Real-Time") \
    .getOrCreate()

# Function to send alert
def send_alert(predicted_attack):
    msg = MIMEText(f"Intrusion Detected: {predicted_attack}")
    msg['Subject'] = 'IDS Alert'
    msg['From'] = 'ids@yourcompany.com'
    msg['To'] = 'admin@yourcompany.com'

    server = smtplib.SMTP('localhost')
    server.sendmail('ids@yourcompany.com', ['admin@yourcompany.com'], msg.as_string())
    server.quit()

# Function to preprocess incoming data
def preprocess_data(df):
    # Implement preprocessing logic here
    processed_df = df  # Replace this with actual preprocessing logic
    return processed_df

# Define schema for incoming JSON data
schema_string = StructType([
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True),
    StructField("feature3", DoubleType(), True),
    # Add more fields based on your incoming Kafka message structure
])

# Stream network data from Kafka
df_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "network_traffic_topic") \
    .load()

# Cast Kafka message value to String and parse it as JSON
df_stream = df_stream.selectExpr("CAST(value AS STRING)")

# Preprocess the incoming data stream using the schema
processed_stream = df_stream \
    .select(col("value").cast(StringType()).alias("json")) \
    .select(from_json(col("json"), schema_string).alias("data"))

# Preprocess the stream
processed_stream = preprocess_data(processed_stream)

# Write the stream to a temporary view for querying
query = processed_stream.writeStream \
    .queryName("processed_stream") \
    .outputMode("append") \
    .format("memory") \
    .start()

# Continuous loop to process the stream
while True:
    predictions_df = spark.sql("SELECT * FROM processed_stream")
    pandas_df = predictions_df.toPandas()

    if not pandas_df.empty:
        # Assuming your features are in columns like feature1, feature2, etc.
        X_new = np.array(pandas_df[['feature1', 'feature2', 'feature3']])  # Adjust based on actual features
        X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))

        try:
            predictions = model.predict(X_new)
            predicted_classes = np.argmax(predictions, axis=1)

            # Map predicted classes to attack categories
            attack_categories = label_encoder.inverse_transform(predicted_classes)

            # Alerting mechanism
            for attack in attack_categories:
                if attack in ["DoS", "Probe"]:
                    send_alert(attack)
        except Exception as e:
            print(f"Error during prediction: {e}")

    time.sleep(5)

spark.stop()
