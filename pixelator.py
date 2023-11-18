import cv2
import os
import numpy as np

class Pixelator:
    def __init__(self, input_dir: str, output_dir: str) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def pixelate_images(self, percentage: int, blur_amount=9, resize_percentage=100) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                input_image_path = os.path.join(self.input_dir, filename)
                pixelated_image = self.pixelate(input_image_path, percentage, blur_amount=blur_amount)

                output_filename = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_filename, pixelated_image)

    def pixelate(self, image_path, percentage: int, blur_amount=9, resize_percentage=100):
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        
        if percentage < 0 or percentage > 100:
            print(f"Invalid percentage: {percentage}")
            return img
        
        img = self._sharpen_image(img)

        # Get the dimensions of the image
        height, width = img.shape[:2]

        block_size_x = int(percentage*width/100)
        block_size_y = int(percentage*height/100)

        # Apply Gaussian blur to the smaller image to enhance edges
        if blur_amount > 0:
            blurred_img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
        else:
            blurred_img = img

        # Resize the image to a smaller size using bilinear interpolation
        small_img = cv2.resize(blurred_img, (block_size_x, block_size_y), interpolation=cv2.INTER_LINEAR)

        resized_height = int(height * resize_percentage/100) 
        resized_width = int(width * resize_percentage/100) 
        
        # Resize the blurred image back to the original size using nearest-neighbor interpolation
        pixelated = cv2.resize(small_img, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def _sharpen_image(self, img):
        # Create a sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # Apply the sharpening kernel
        sharpened_img = cv2.filter2D(img, -1, kernel)

        return sharpened_img
        # filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=10, sigmaSpace=10)

        # return filtered_img

    def kmeans(self, pixel_size: int, upscale_percentage=200, num_clusters=32, blur=3) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                input_image_path = os.path.join(self.input_dir, filename)
                print(f"Processing {input_image_path}")
                
                image = self.pixelate_and_segment(input_image_path, pixel_size, num_clusters, blur, upscale_percentage)

                output_filename = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_filename, image)
    
    def pixelate_and_segment(self, image_path, pixel_size, num_clusters, blur, upscale):
        # Read the image
        image = cv2.imread(image_path)
        
        # Pixelate the image
        small = cv2.GaussianBlur(image, (blur, blur), 0)
        small = cv2.resize(small, None, fx=1.0 / pixel_size, fy=1.0 / pixel_size, interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(small, (int(image.shape[1]*upscale), int(image.shape[0]*upscale)), interpolation=cv2.INTER_NEAREST)

        # Reshape the image into a 2D array of pixels
        pixels = result.reshape((-1, 3))  # Reshape to a 2D array (rows, columns)

        # Convert to float type for k-means processing
        pixels = np.float32(pixels)

        # Define criteria for k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.3)

        # Perform k-means clustering
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to uint8
        centers = np.uint8(centers)

        # Map each pixel to its corresponding center color
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(result.shape)

        return segmented_image   
