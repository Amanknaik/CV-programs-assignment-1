import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path='image.jpg'
img = cv2.imread(image_path,0)

# if image is None:
#     print(f"Could not open or find the image at '{image_path}")

# else:
#     histogram = cv2.calcHist([image],[0],None,[256],[0,256])

#     plt.figure(figsize=(8,6))
#     plt.title('Image Histogram')
#     plt.xlabel('Image Histogram')
#     plt.ylabel('Frequency')
#     plt.plot(histogram, color='black')
#     plt.xlim([0,256])
#     plt.grid(True)
#     plt.show()

w = img.shape[0]
h = img.shape[1]
frequencies = list(np.zeros(256, dtype= 'int32'))

#for each pixel in image array, increment the corresponding intensity level in intensities
for i in range(w):
  for j in range(h):
    frequencies[img[i, j]] += 1

for i, j in enumerate(frequencies):
  print(f'{i} : {j}')

#initialise an array to store the labels of all intensities(i.e array like [0, 1, 2, 3......., 255])
intensity_levels = np.arange(len(frequencies), step= 1)
print(intensity_levels)

#plot and display a histogram of intensity levels vs frequencies of occurence
plt.hist(frequencies, bins= intensity_levels, density= True, color="black")
plt.xlabel("Intensity level")
plt.ylabel("Frequency")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()