import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'

import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.data_loader import create_dataset_dataframe, TRAIN_DIR, TEST_DIR
from src.tensorflow.dataset import create_tf_dataset
from src.tensorflow.model import create_simple_cnn

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def main():
  """
  Main function to train and evaluate the TensorFlow CNN.
  """
  print("Loading dataframes...")
  train_df = create_dataset_dataframe(str(TRAIN_DIR))
  test_df = create_dataset_dataframe(str(TEST_DIR))
  
  classes = sorted(train_df['label'].unique())
  class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
  
  print("Creating datasets...")
  train_dataset = create_tf_dataset(train_df, class_to_idx, IMAGE_SIZE, BATCH_SIZE, is_training=True)
  test_dataset = create_tf_dataset(test_df, class_to_idx, IMAGE_SIZE, BATCH_SIZE, is_training=False)
  
  model = create_simple_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=len(classes))
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
  
  print("\nStarting training...")
  start_time = time.time()
  
  model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1
  )
  
  training_time = time.time() - start_time
  print(f"\nTraining completed in {training_time:.2f} seconds.")
  
  print("\nEvaluating on test set...")
  start_time = time.time()
  
  y_pred_logits = model.predict(test_dataset)
  y_pred = np.argmax(y_pred_logits, axis=1)
  
  inference_time = time.time() - start_time
  
  y_true = test_df['label'].map(class_to_idx).values
  
  accuracy = accuracy_score(y_true, y_pred)
  print(f"Inference completed in {inference_time:.2f} seconds.")
  print(f"\nTensorFlow CNN Accuracy: {accuracy * 100:.2f}%")
  
  print("\nClassification Report:")
  print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
  main()
