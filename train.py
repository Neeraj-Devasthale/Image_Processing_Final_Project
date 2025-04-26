import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model

def train_model(model, num_epochs, train_loader, val_loader, loss_fn, optimizer):
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=num_epochs,
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    
    
    return history