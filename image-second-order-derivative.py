import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# if image is None:
#     print(f"Could not open or find the image at '{image_path}")

# else:
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)

#     plt.figure(figsize=(10,5))

#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(laplacian, cmap='gray')
#     plt.title('Laplacian (2nd Order Derivative)')
#     plt.axis('off')

#     plt.show()

#Horizontal gradient
Ix=np.zeros_like(image, dtype=np.float64)
for i in range(image.shape[0]):
    Ix[i,1:-1]=image[i,2:]-image[i,:-2]

#Vertical gradient
Iy=np.zeros_like(image, dtype=np.float64)
for i in range(image.shape[1]):
    Iy[1:-1,i]=image[2:,i]-image[:-2,i]

# Compute the second-order partial derivatives
Ixx = np.zeros_like(Ix, dtype=np.float64)
for i in range(Ix.shape[0]):
    Ixx[i, 1:-1] = Ix[i, 2:] - Ix[i, :-2]

Iyy = np.zeros_like(Iy, dtype=np.float64)
for i in range(Iy.shape[1]):
    Iyy[1:-1, i] = Iy[2:, i] - Iy[:-2, i]

Ixy = np.zeros_like(Ix, dtype=np.float64)
for i in range(Ix.shape[0] - 2):
    Ixy[i + 1, 1:-1] = Ix[i + 2, 1:-1] - Ix[i, 1:-1]

#Displaying the original image and the horizontal and vertical gradient
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(Ix, cmap='gray')
plt.title("Horizontal Gradient")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(Iy, cmap='gray')
plt.title("Vertical Gradient")
plt.axis('off')

plt.show()
