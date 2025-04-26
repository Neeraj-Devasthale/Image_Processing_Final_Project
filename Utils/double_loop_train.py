import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python.util import deprecation

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Suppress Keras format warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*HDF5 file.*')


def train_model(model, num_epochs, train_loader, val_loader, loss_fn, optimizer):
    # Initialize metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

    # History tracking
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 15
    wait = 0

    for epoch in range(num_epochs):
        # Reset metrics (corrected method)
        train_loss.reset_state()
        train_acc.reset_state()
        val_loss.reset_state()
        val_acc.reset_state()

        # Training loop
        for x_train, y_train in train_loader:
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = loss_fn(y_train, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_acc(y_train, predictions)

        # Validation loop
        for x_val, y_val in val_loader:
            val_preds = model(x_val, training=False)
            v_loss = loss_fn(y_val, val_preds)
            val_loss(v_loss)
            val_acc(y_val, val_preds)

        # Store history
        history['loss'].append(train_loss.result().numpy())
        history['accuracy'].append(train_acc.result().numpy())
        history['val_loss'].append(val_loss.result().numpy())
        history['val_accuracy'].append(val_acc.result().numpy())

        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss.result():.4f}, '
              f'Train Acc: {train_acc.result():.4f}, '
              f'Val Loss: {val_loss.result():.4f}, '
              f'Val Acc: {val_acc.result():.4f}')

        # Early stopping check
        # Inside the early stopping check
        if val_loss.result() < best_val_loss:
           best_val_loss = val_loss.result()
           wait = 0
           # Save best weights with correct extension
           model.save_weights('checkpoints/best_weights.weights.h5')
        else:
           wait += 1
           if wait >= patience:
               print(f'Early stopping at epoch {epoch+1}')
               break

       # At the end of training
        model.load_weights('checkpoints/best_weights.weights.h5')
        model.save('checkpoints/final_model.keras')  # Save full model

    # Load best weights
    model.load_weights('best_weights.h5')

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()
    
    return history