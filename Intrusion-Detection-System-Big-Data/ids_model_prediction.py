import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IDS LSTM Prediction") \
    .getOrCreate()

# Load the test dataset (as you don't need to retrain)
test_df = spark.read.parquet("/home/kali/Bg01/Dataset/UNSW-NB15/UNSW_NB15_testing-set.parquet")

# Data preprocessing (same as training preprocessing)
def preprocess_data(df):
    string_columns = ["proto", "service", "state", "attack_cat"]
    for column in string_columns:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index")
        df = indexer.fit(df).transform(df)

    # Assemble features
    feature_columns = [f"{col}_index" for col in string_columns] + [col for col in df.columns if col not in string_columns and col != 'label']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    return df.select("features", "attack_cat")

test_df = preprocess_data(test_df)

# Convert Spark DataFrame to Pandas
test_pandas = test_df.toPandas()

# Prepare the data for LSTM
def prepare_data(df):
    X = np.array(list(df['features']))
    y = np.array(df['attack_cat'])
    return X, y

X_test, y_test = prepare_data(test_pandas)

# Convert y_test to numeric values
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)

# Reshape input to be [samples, time steps, features]
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Load the saved LSTM model
model = load_model("/home/kali/Bg01/saved_model/enhanced_lstm_model.keras")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Decode predicted labels back to their original attack categories
decoded_predictions = label_encoder.inverse_transform(predicted_classes)

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Actual': label_encoder.inverse_transform(y_test),
    'Predicted': decoded_predictions
})
predictions_df.to_csv("/home/kali/Bg01/saved_model/predictions.csv", index=False)

# Evaluate the model (accuracy on test data)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Stop Spark session
spark.stop()

# Plotting a confusion matrix to visualize predictions vs actual
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(label_encoder.inverse_transform(y_test), decoded_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix - IDS Predictions")
plt.savefig("/home/kali/Bg01/saved_model/confusion_matrix.png")
plt.show()
