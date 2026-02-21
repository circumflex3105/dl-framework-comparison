import tensorflow as tf

def create_simple_cnn(input_shape: tuple = (150, 150, 3), num_classes: int = 6) -> tf.keras.Model:
  """
  Creates a simple Convolutional Neural Network using Keras Sequential API.
  """
  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes)
  ])
  
  return model
