import os
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io
from skimage.transform import resize

directory_unlabeled = 'D:/Facultate/Disertatie/mainProject/pythonProject1/organized_labeled_ultrasound_dataset/unlabeled_train'
directory_labeled = 'D:/Facultate/Disertatie/mainProject/pythonProject1/organized_labeled_ultrasound_dataset/labeled_train'

labeled_by_class = {}
for root_l, _, files_l in os.walk(directory_labeled):
    current_class = os.path.basename(root_l)
    if not files_l: continue
    labeled_by_class[current_class] = files_l

files = {}
names_classes = list(labeled_by_class.keys())
num_classes = len(names_classes)
correct_predictions = 0
total_predictions = 0
batch_size = 120
batch_per_class = batch_size // num_classes

for root_u, _, files_u in os.walk(directory_unlabeled):
    for filename_u in files_u:
        classes = {name: 0.0 for name in names_classes}
        true_label = filename_u.split('_')[0].lower()

        print(f"Processing: {filename_u}")
        path_img_u = os.path.join(root_u, filename_u)
        img_u = io.imread(path_img_u, as_gray=True)

        batch = []
        for c_name in names_classes:
            labeled_files = labeled_by_class[c_name]
            chosen = np.random.choice(labeled_files, size=batch_per_class, replace=True)
            for f_name in chosen:
                batch.append((f_name, c_name))

        for f_name, c_name in batch:
            # print(sample)
            path_img_l = os.path.join(directory_labeled, c_name, f_name)
            img_l = io.imread(path_img_l, as_gray=True)
            img_l = resize(img_l, img_u.shape, anti_aliasing=True)
            val_ssim = ssim(img_u, img_l, data_range=img_u.max() - img_u.min())
            classes[c_name] += val_ssim

        body_part = max(classes, key=classes.get)
        files[filename_u] = body_part
        total_predictions += 1

        if body_part in true_label:
            correct_predictions += 1

        print(classes)
        print(f"Predicted: {body_part}")

accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
df = pd.DataFrame(list(files.items()), columns=['Filename', 'Predicted Body Part'])
df.to_excel('ssim_results.xlsx', index=False)

print(f"Excel file saved successfully.")
print(f"Total Images: {total_predictions}")
print(f"Correct: {correct_predictions}")
print(f"Overall Accuracy: {accuracy:.2f}%")