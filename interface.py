# replace MyCustomModel with the name of your model
from model import build_model as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import train_model as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import test as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import Dataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import data_loader as the_dataloader


## OTHER CUSTOM CONFIG IMPORTS
from config import batch_size as the_batch_size
from config import num_epochs as total_epochs
from config import resize_x, resize_y, input_channels, classes
from config import loss_fn as loss_fn
from config import optimizer as optimizer
import tensorflow as tf