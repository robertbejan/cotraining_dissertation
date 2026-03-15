import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt

# 1. Load the actual image (Use PIL for .png files)
img_path = "D:/Facultate/Disertatie/mainProject/pythonProject1/small_labeled_ultrasound_dataset/labeled_train/abdomen/Patient00704_Plane2_1_of_1.png"
image = Image.open(img_path).convert('L') # Load as Grayscale

# 2. Transform to Tensor
gray_tensor = transforms.ToTensor()(image)

# 3. Apply the FFT Logic
f_complex = torch.fft.fft2(gray_tensor.squeeze())
f_shifted = torch.fft.fftshift(f_complex)

# Magnitude and Log Scale
fft_magnitude = torch.abs(f_shifted)
fft_log = torch.log1p(fft_magnitude)

# Normalize for visualization
fft_viz = (fft_log - fft_log.min()) / (fft_log.max() - fft_log.min() + 1e-8)

# 4. Visualization
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Original Ultrasound (Spatial)")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("FFT Log-Magnitude (Frequency)")
plt.imshow(fft_viz.numpy(), cmap='magma') # Magma helps see the energy distribution
plt.axis('off')

plt.show()