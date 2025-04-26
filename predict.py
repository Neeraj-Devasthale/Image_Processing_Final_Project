# _predict.py
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from dataset import Dataset
from config import classes as class_names

def test(model, test_loader=None, test_dir='data'):
    # Initialize dataset and load test data
    test_data = Dataset(data_dir='', test_dir=test_dir)  # Empty data_dir for training data
    try:
        X_test, y_test, filenames = test_data.load_test_data()
        print(f"✅ Loaded {len(X_test)} test samples")
    except Exception as e:
        print(f"❌ Error loading test data: {str(e)}")
        return pd.DataFrame()

    # Create test loader if not provided
    if test_loader is None:
        test_loader = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test)).batch(256)

    # Prediction loop
    y_true = []
    y_pred = []
    
    for batch, (x, y) in enumerate(test_loader):
        preds = model.predict(x)
        batch_preds = np.argmax(preds, axis=1)
        
        y_true.extend(y.numpy())
        y_pred.extend(batch_preds)

    # Create results DataFrame
    results = pd.DataFrame({
        'filename': filenames,
        'predicted_class': [class_names[p] for p in y_pred],
        'true_class': [class_names[true] for true in y_true]
    })

    # Save report
    csv_path = os.path.join(test_dir, 'predictions_report.csv')
    results.to_csv(csv_path, index=False)
    print(f"💾 Saved predictions to {csv_path}")

    # Generate classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    print("\n🎯 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(tabulate(cm, headers=class_names, showindex=class_names))

    return results