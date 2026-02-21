import tensorflow as tf
import pandas as pd

def load_and_preprocess_image(filepath: str, label: int, image_size: tuple):
  """
  Loads and preprocesses an image from a filepath.
  """
  image = tf.io.read_file(filepath)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, image_size)
  
  image = tf.cast(image, tf.float32) / 255.0
  
  mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
  std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
  image = (image - mean) / std
  
  return image, label

def create_tf_dataset(df: pd.DataFrame, class_to_idx: dict, image_size: tuple, batch_size: int, is_training: bool = False) -> tf.data.Dataset:
  """
  Creates a tf.data.Dataset from a pandas DataFrame.
  """
  filepaths = df['filepath'].values
  labels = df['label'].map(class_to_idx).values
  
  dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
  
  if is_training:
    dataset = dataset.shuffle(buffer_size=len(df))
    
  dataset = dataset.map(
    lambda x, y: load_and_preprocess_image(x, y, image_size),
    num_parallel_calls=tf.data.AUTOTUNE
  )
  
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  
  return dataset
