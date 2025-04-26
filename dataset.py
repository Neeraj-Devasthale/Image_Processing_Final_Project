import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class Dataset:
    def __init__(self, data_dir, processed_dir='processed_data', test_dir='data'):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.test_dir = test_dir  # Separate directory for test files
        self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'm']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def load_test_data(self):
        """Load and process test data from separate test directory"""
        print(f"Loading test data from: {self.test_dir}")
        test_data = []
        test_labels = []
        test_filenames = []
        valid_files = 0

        for filename in os.listdir(self.test_dir):
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(self.test_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                # Validate test sample
                mfccs = json_data.get('mfccs', [])
                if not mfccs or np.array(mfccs).shape != (39, 120):
                    continue

                # Process test data
                label = filename[0]
                if label not in self.class_to_idx:
                    continue

                spectrogram = np.array(mfccs)
                spectrogram = np.expand_dims(spectrogram, axis=-1)
                test_data.append(spectrogram)
                test_labels.append(self.class_to_idx[label])
                test_filenames.append(filename)
                valid_files += 1

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        # Convert to numpy arrays
        X_test = np.array(test_data)
        y_test = np.array(test_labels)
        
        print(f"Loaded {valid_files} valid test samples")
        return X_test, y_test, test_filenames

    def load_data(self, reprocess=False):
        if not reprocess and os.path.exists(os.path.join(self.processed_dir, 'X_train.npy')):
            return self.load_processed_data()
            
        print(f"Loading data from: {self.data_dir}")
        valid_files = 0
        self.data = []
        self.labels = []
        self.filenames = []
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.json'): continue
                
            file_path = os.path.join(self.data_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Validate data
                mfccs = json_data.get('mfccs', [])
                if not mfccs or np.array(mfccs).shape != (39, 120):
                    continue
                
                # Validate label
                label = filename[0]
                if label not in self.class_to_idx:
                    continue
                
                # Process data
                spectrogram = np.array(mfccs)
                spectrogram = np.expand_dims(spectrogram, axis=-1)
                self.data.append(spectrogram)
                self.labels.append(self.class_to_idx[label])
                valid_files += 1
                
            except Exception as e:
                continue

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.filenames.append(filename)
        print(f"Loaded {valid_files} valid samples")

        # Split into train/validation only
        X_train, X_val, y_train, y_val = self.split_data()
        self.save_processed_data(X_train, X_val, y_train, y_val)
        
        return X_train, X_val, y_train, y_val

    def split_data(self, validation_size=0.2):
        """Only split into train/validation"""
        return train_test_split(
            self.data, self.labels,
            test_size=validation_size,
            random_state=42
        )

    def save_processed_data(self, X_train, X_val, y_train, y_val):
        np.save(os.path.join(self.processed_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(self.processed_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(self.processed_dir, 'classes.npy'), np.array(self.classes))

    def load_processed_data(self):
        X_train = np.load(os.path.join(self.processed_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(self.processed_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(self.processed_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(self.processed_dir, 'y_val.npy'))
    
        return X_train, X_val, y_train, y_val



def load_data(self, reprocess=False):
    if not reprocess and os.path.exists(os.path.join(self.processed_dir, 'X_train.npy')):
        return self.load_processed_data()


def data_loader(data_dir, batch_size=256, mode='train'):
    """Unified data loader function"""
    dataset = Dataset(data_dir)
    
    if mode == 'train':
        # Get train/val splits
        X_train, X_val, y_train, y_val = dataset.load_data()
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        return train_ds, val_ds
        
    elif mode == 'test':
        # Load test data from separate directory
        X_test, y_test, _ = dataset.load_test_data()
        return tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        
    elif mode == 'inference':
        # For prediction on new data
        dataset.load_data(is_prediction=True)
        return tf.data.Dataset.from_tensor_slices(dataset.data).batch(batch_size)
        
    else:
        raise ValueError("Invalid mode. Choose: 'train', 'test', or 'inference'")