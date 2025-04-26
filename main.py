# main.py
from interface import *
import tensorflow as tf

if __name__ == "__main__":
    # Training workflow
    train_loader, val_loader = the_dataloader(
        data_dir='JSONS',
        batch_size=the_batch_size,
        mode='train'
    )
    
    input_shape = (resize_x, resize_y, input_channels)
    model = TheModel(input_shape, len(classes))
    
    the_trainer(
        model=model,
        num_epochs=total_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    
    model.save('checkpoints/final_weights.h5')
    print("✅ Model saved to checkpoints/final_weights.h5")
    
    test_loader = the_dataloader(
        data_dir='data',
        batch_size=the_batch_size,
        mode='test'
    )
    
    the_predictor(model, test_loader=test_loader)
