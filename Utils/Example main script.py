# interface.py
from model import build_model as TheModel
from train import train_model as the_trainer
from predict import test as the_predictor
from dataset import Dataset as TheDataset
from dataset import data_loader as the_dataloader  
from config import batch_size as the_batch_size
from config import num_epochs as total_epochs
from config import resize_x, resize_y, input_channels, classes
from config import loss_fn as loss_fn
from config import optimizer as optimizer
import tensorflow as tf

if __name__ == "__main__":
    # Training workflow
    # Get train and validation loaders using unified data loader
    train_loader, val_loader = the_dataloader(
        data_dir='JSONS',
        batch_size=the_batch_size,
        mode='train'
    )
    
    # Initialize model
    input_shape = (resize_x, resize_y, input_channels)
    model = TheModel(input_shape, len(classes))
    
    # Train model
    the_trainer(
        model=model,
        num_epochs=total_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    
    # Save final model weights
    model.save('checkpoints/final_weights.h5')
    print("✅ Model saved to checkpoints/final_weights.h5")
    
    # Test evaluation
    # Get test loader using same data loader
    test_loader = the_dataloader(
        data_dir='data',
        batch_size=the_batch_size,
        mode='test'
    )
    
    # Generate and save predictions
    the_predictor(model, test_loader=test_loader)