import os
import random
import shutil

# Update with your actual class labels
CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'm']
SAMPLES_PER_CLASS = 10

def move_test_samples(source_dir='JSONS', dest_dir='data'):
    # Create directories if needed
    os.makedirs(dest_dir, exist_ok=True)
    
    # Organize files by class
    class_files = {cls: [] for cls in CLASSES}
    
    # Group files by class
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            class_label = filename[0].lower()
            if class_label in class_files:
                class_files[class_label].append(filename)
    
    # Move samples for each class
    moved_count = 0
    for cls, files in class_files.items():
        if len(files) < SAMPLES_PER_CLASS:
            print(f"Warning: Class {cls} only has {len(files)} files. Moving all available.")
        
        # Select files to move
        selected = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        
        # Move selected files
        for f in selected:
            src = os.path.join(source_dir, f)
            dest = os.path.join(dest_dir, f)
            shutil.move(src, dest)
            moved_count += 1
            
    print(f"Moved {moved_count} test samples ({SAMPLES_PER_CLASS} per class) to {dest_dir}")

if __name__ == "__main__":
    move_test_samples()