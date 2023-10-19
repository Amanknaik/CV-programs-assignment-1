import cv2
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image=cv2.imread(image_path,0)

if image is None:
    print(f"Could not open or find the image at '{image_path}")

else:
    equalized_image = cv2.equalizeHist(image)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    plt.show()
