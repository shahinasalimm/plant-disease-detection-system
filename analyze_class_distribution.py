import os

# Count the number of images in each class
train_dir = "Dataset/train"
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}

print("Class Distribution in Training Set:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")
