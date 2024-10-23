import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IDS LSTM Enhanced") \
    .getOrCreate()

# Load datasets
train_df = spark.read.parquet("/home/kali/Bg01/Dataset/UNSW-NB15/UNSW_NB15_training-set.parquet")
test_df = spark.read.parquet("/home/kali/Bg01/Dataset/UNSW-NB15/UNSW_NB15_testing-set.parquet")

# Data preprocessing function
def preprocess_data(df):
    string_columns = ["proto", "service", "state", "attack_cat"]
    for column in string_columns:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index")
        df = indexer.fit(df).transform(df)

    feature_columns = [f"{col}_index" for col in string_columns] + [col for col in df.columns if col not in string_columns and col != 'label']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    return df.select("features", "attack_cat")

# Preprocess the train and test data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Convert Spark DataFrames to Pandas DataFrames
train_pandas = train_df.toPandas()
test_pandas = test_df.toPandas()

# Prepare data for LSTM
def prepare_data(df):
    X = np.array(list(df['features']))
    y = np.array(df['attack_cat'])
    return X, y

X_train, y_train = prepare_data(train_pandas)
X_test, y_test = prepare_data(test_pandas)

# Label encoding the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Split the data into 50/50 training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Build an enhanced LSTM model with Bidirectional layers and Batch Normalization
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train_split.shape[1], X_train_split.shape[2])))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Multi-class classification

# Compile the model with a modified optimizer and loss function
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model on the split training/validation set
history = model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), 
                    epochs=100, batch_size=128, callbacks=[early_stopping])

# Save the enhanced model
model.save("/home/kali/Bg01/saved_model/enhanced_lstm_model.keras")

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv("/home/kali/Bg01/saved_model/enhanced_training_history.csv", index=False)

# Plot accuracy and loss graphs with more detailed formatting
plt.figure(figsize=(16, 6))

# Plotting Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Enhanced Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Enhanced Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Save the new plot as a PNG file
plt.tight_layout()
plt.savefig("/home/kali/Bg01/saved_model/enhanced_training_plot.png")

# Stop Spark session
spark.stop()
