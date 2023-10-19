import cv2
import numpy as np

image_path='image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Could not open or find the image at '{image_path}")
else:
    image=cv2.resize(image,(2000,2000))
    edges = cv2.Canny(image, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image= cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
    cv2.drawContours(result_image, contours, -1, (0,255,0), 2)

    cv2.imshow('Original Image', image)
    cv2.imshow('Object Borders', result_image)

    cv2.waitKey(0)