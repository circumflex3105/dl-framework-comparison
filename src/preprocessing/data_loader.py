import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).parent.parent.parent
TRAIN_DIR = BASE_DIR / "data" / "seg_train" / "seg_train"
TEST_DIR = BASE_DIR / "data" / "seg_test" / "seg_test"

def create_dataset_dataframe(data_dir: str) -> pd.DataFrame:
  """
  Creates a pandas DataFrame containing file paths and labels for the dataset.
  """
  data = []
  data_path = Path(data_dir)
  
  for class_dir in data_path.iterdir():
    if not class_dir.is_dir():
      continue
      
    label = class_dir.name
    for img_path in class_dir.glob('*.jpg'):
      data.append({
        'filepath': str(img_path),
        'label': label
      })
              
  df = pd.DataFrame(data)
  return df

def explore_dataset(df: pd.DataFrame, title: str):
  """
  Prints basic statistics and plots class distribution.
  """
  print(f"--- {title} ---")
  print(f"Total images: {len(df)}")
  print("\nClass distribution:")
  class_counts = df['label'].value_counts()
  print(class_counts)
  
  plt.figure(figsize=(10, 6))
  class_counts.plot(kind='bar')
  plt.title(f'Class Distribution - {title}')
  plt.xlabel('Class')
  plt.ylabel('Number of Images')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

def visualize_samples(df: pd.DataFrame, num_samples: int = 5):
  """
  Visualizes a few random samples from each class.
  """
  classes = df['label'].unique()
  fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 3 * len(classes)))
  
  for i, cls in enumerate(classes):
    sample_df = df[df['label'] == cls].sample(n=num_samples, random_state=42)
    
    for j, (_, row) in enumerate(sample_df.iterrows()):
      img = Image.open(row['filepath'])
      axes[i, j].imshow(img)
      axes[i, j].axis('off')
      if j == 0:
        axes[i, j].set_title(cls, loc='left', pad=10)
              
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  print("Loading datasets...")
  train_df = create_dataset_dataframe(str(TRAIN_DIR))
  test_df = create_dataset_dataframe(str(TEST_DIR))
  
  explore_dataset(train_df, "Training Set")
  explore_dataset(test_df, "Test Set")
  
  print("\nVisualizing samples from the training set...")
  visualize_samples(train_df)
