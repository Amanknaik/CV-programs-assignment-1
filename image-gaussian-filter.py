import cv2
import numpy as np
image_path='image.jpg'
image=cv2.imread(image_path)
import matplotlib.pyplot as plt

# if image is None:
#     print(f"Could not open or find the image at '{image_path}")
# else:
#     width=2000
#     height=2000
#     ksize=(5,5)
#     resized_image = cv2.resize(image, (width, height))
#     blurred_image = cv2.GaussianBlur(resized_image,ksize,0)

#     cv2.imshow('Original Image',resized_image)
#     cv2.imshow('Smoothed Image',blurred_image)

#------------------Gaussian Function-----------------
# Gaussian function is a bell shaped curve that represents a probability distribution.
# the formula for the 1D Gaussian function is:

# G(x) = (1(sqrt(2*pi*(sigma)^2))e^(-(x^2)/2*(sigma)^2)

sigma =1.5

#function to return a 1D Gaussian kernel
def gaussian_kernel(sigma):
    size=int(6*sigma)
    if size%2==0:
        size+=1
    kernel = np.exp(-(np.arange(-size//2,size//2+1)**2)/(2*sigma**2))
    return kernel/kernel.sum()

kernel=gaussian_kernel(sigma)

# Apply horizontal convolution
smoothed_img = np.zeros_like(image, dtype=np.float64)
for i in range(image.shape[0]):
    for j in range(image.shape[1] - len(kernel) + 1):
        kernel_reshaped=kernel.reshape(-1,1)
        smoothed_img[i, j + len(kernel) // 2] = np.sum(image[i, j:j+len(kernel)] * kernel_reshaped)

# Apply vertical convolution
for j in range(smoothed_img.shape[1]):
    for i in range(smoothed_img.shape[0] - len(kernel) + 1):
        kernel_reshaped=kernel.reshape(-1,1)
        smoothed_img[i + len(kernel) // 2, j] = np.sum(smoothed_img[i:i+len(kernel), j] * kernel_reshaped)

# Convert back to uint8 data type (required for saving as an image)
smoothed_img = np.uint8(smoothed_img)

# Display original and smoothed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_img, cmap='gray')
plt.title("Smoothed Image")
plt.axis('off')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()