import os
import random
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from numpy import fft
import cv2


def filtering_text(src):
    img = cv2.imread(src)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    mask = np.zeros_like(gray, dtype=np.uint8)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(mask, [c], 0, 255, -1)

    result1 = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return result1


def fast_fourier_transformation(src):
    # img = cv2.imread(src_path)

    filtered = filtering_text(src)
    gray_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    gray_filtered_resized = cv2.resize(gray_filtered, (227, 227))

    fft_filtered = fft.fft2(gray_filtered_resized)
    print(fft_filtered.size)

    return fft_filtered


def preprocessing_img(src, dest, method='FFT'):
    if method == 'RGB':
        filtered = filtering_text(src)
        filtered = transform(filtered)

        save_image(filtered, dest)
    elif method == 'FFT':
        fft_img = fast_fourier_transformation(src)
        fft_img = transform(fft_img)

        np.save(dest, fft_img)
    else:
        pass

    return None


# Define the base directory containing all the class folders
base_directory = 'D:/Facultate/Disertatie/mainProject/pythonProject1/all images'

# Define the directories for train and testing datasets
train_directory = 'D:/Facultate/Disertatie/mainProject/pythonProject1/trainFFTResized'
test_directory = 'D:/Facultate/Disertatie/mainProject/pythonProject1/testFFTResized'

# Define the ratio for splitting the data (e.g., 80% train, 20% test)
train_ratio = 0.2
crop_left = 30
crop_right = 30

transform = transforms.Compose([
    transforms.ToTensor()  # Converts to tensor and scales pixel values to [0, 1]
])

# Iterate over each class folder
for class_folder in os.listdir(base_directory):
    class_path = os.path.join(base_directory, class_folder)

    # Ensure it's a directory
    if os.path.isdir(class_path):
        # Get all images in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Shuffle the images randomly
        random.shuffle(images)

        # Split the images into train and testing sets
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Create class-specific directories in train and test folders
        train_class_dir = os.path.join(train_directory, class_folder)
        test_class_dir = os.path.join(test_directory, class_folder)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Move images to the respective directories
        for image in train_images:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(train_class_dir, image)

            preprocessing_img(src_path, dest_path)

            # src_path = os.path.join(class_path, image)
            # dest_path = os.path.join(train_class_dir, image)
            # shutil.move(src_path, dest_path)

        for image in test_images:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(test_class_dir, image)

            preprocessing_img(src_path, dest_path)

            # src_path = os.path.join(class_path, image)
            # dest_path = os.path.join(test_class_dir, image)
            # shutil.move(src_path, dest_path)

print("Data split and moved to train and test directories.")
