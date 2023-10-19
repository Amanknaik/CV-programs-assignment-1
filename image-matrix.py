import cv2

image_path='image.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Could not open or find the image at '{image_path}")

else:
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Image Matrix:")
    print(image)
    