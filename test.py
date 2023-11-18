import cv2
import numpy as np

def pixelate(image, pixel_size):
    # Resize the image to a smaller size
    small = cv2.resize(image, None, fx=1.0/pixel_size, fy=1.0/pixel_size, interpolation=cv2.INTER_NEAREST)

    # Enlarge the small image to the original size
    result = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return result

def edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # blurred = gray

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 20, 100)

    return edges

# Read the image
input_image = cv2.imread('input_folder/shuttle.png')

# Perform edge detection
edges = pixelate(edge_detection(input_image), 2)
cv2.imshow('Edges', edges)

# Pixelate the image (adjust pixel size as needed)
pixelated_image = pixelate(input_image, 2)  # Change the pixel size as needed
cv2.imshow('Pixelated', pixelated_image)

# Merge the pixelated image with the edges to emphasize edges
result = cv2.bitwise_and(pixelated_image, pixelated_image, mask=edges)
print(pixelated_image)
print(edges)
# result = cv2.bitwise_or(pixelated_image, result)
mask = (edges == 0).astype(np.uint8)
print(mask)
pixelated_image[mask == 0] = [0,0,0]
# Display the result
cv2.imshow('Pixel Art with Sharp Edges', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
