import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# if image is None:
#     print(f"Could not open or find the image at '{image_path}")

# else:
#     sobel_x = cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize=3)
    
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1,3,1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.imshow(sobel_x, cmap='gray')
#     plt.title('Sobel X (1st Order Derivative)')
#     plt.axis('off')
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(sobel_y, cmap='gray')
#     plt.title('Sobel Y (1st Order Derivative)')
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