import cv2
import numpy as np

def pixelate(image, pixel_size):
    # Resize the image to a smaller size
    small = cv2.GaussianBlur(image, (3,3), 0)
    small = cv2.resize(small, None, fx=1.0/pixel_size, fy=1.0/pixel_size, interpolation=cv2.INTER_NEAREST)


    # Enlarge the small image to the original size
    result = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return result

# Read the image
image = cv2.imread('input_folder/car.jpeg')

image = pixelate(image, 3)

# Reshape the image into a 2D array of pixels
pixels = image.reshape((-1, 3))  # Reshape to a 2D array (rows, columns)

# Convert to float type for k-means processing
pixels = np.float32(pixels)

# Define criteria for k-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)

# Number of clusters (adjust this based on the desired reduction)
num_clusters = 32

# Perform k-means clustering
_, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to uint8
centers = np.uint8(centers)

# Map each pixel to its corresponding center color
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Display or save the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# If you want to save the segmented image:
# cv2.imwrite('segmented_image.jpg', segmented_image)
