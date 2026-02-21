import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.data_loader import create_dataset_dataframe, TRAIN_DIR, TEST_DIR

IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42

def load_and_flatten_images(df: pd.DataFrame) -> np.ndarray:
  """
  Loads images from filepaths, resizes them, and flattens them into 1D arrays.
  """
  features = []
  total_images = len(df)
  
  for idx, row in df.iterrows():
    if idx > 0 and idx % 2000 == 0:
      print(f"Processed {idx}/{total_images} images...")
      
    try:
      img = Image.open(row['filepath'])
      img = img.resize(IMAGE_SIZE)
      img_array = np.array(img)
      
      if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        img = img.convert('RGB')
        img_array = np.array(img)
        
      flattened = img_array.flatten()
      features.append(flattened)
    except Exception as e:
      print(f"Error loading image {row['filepath']}: {e}")
      features.append(np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3))
      
  return np.array(features)

def train_and_evaluate_baseline():
  """
  Trains a Random Forest classifier on the flattened images and evaluates it.
  """
  print("Loading dataset dataframes...")
  train_df = create_dataset_dataframe(str(TRAIN_DIR))
  test_df = create_dataset_dataframe(str(TEST_DIR))
  
  print(f"Training set size: {len(train_df)}")
  print(f"Test set size: {len(test_df)}")
  
  print("\nEncoding labels...")
  label_encoder = LabelEncoder()
  y_train = label_encoder.fit_transform(train_df['label'])
  y_test = label_encoder.transform(test_df['label'])
  
  print(f"\nExtracting features (resizing to {IMAGE_SIZE} and flattening)...")
  print("Processing training images:")
  X_train = load_and_flatten_images(train_df)
  print("Processing test images:")
  X_test = load_and_flatten_images(test_df)
  
  print(f"\nFeature matrix shape: {X_train.shape}")
  
  print("\nTraining Random Forest baseline model...")
  start_time = time.time()
  
  clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
  clf.fit(X_train, y_train)
  
  training_time = time.time() - start_time
  print(f"Training completed in {training_time:.2f} seconds.")
  
  print("\nEvaluating model on test set...")
  start_time = time.time()
  y_pred = clf.predict(X_test)
  inference_time = time.time() - start_time
  
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Inference completed in {inference_time:.2f} seconds.")
  print(f"\nBaseline Accuracy: {accuracy * 100:.2f}%")
  
  print("\nClassification Report:")
  target_names = label_encoder.classes_
  print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
  train_and_evaluate_baseline()
