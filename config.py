# Model parameters
resize_x = 39
resize_y = 120
input_channels = 1
classes = ['a', 'b', 'c', 'd', 'e', 'f', 'm']

# Training parameters
batch_size = 512
num_epochs = 200
learning_rate = 0.0001

# Loss and optimizer
import tensorflow as tf
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)