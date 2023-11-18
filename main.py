import cv2
import os

from pixelator import Pixelator

input_dir = 'input_folder'
output_dir = 'output_folder'

pix = Pixelator(input_dir, output_dir)

# pix.pixelate_images(20, blur_amount=3, resize_percentage=200)
# pix.kmeans(pixel_size=3, upscale_percentage=200, num_clusters=32, blur=3)

# pix.pixelate_and_segment('input_folder/car.jpeg',3,32)
pix.kmeans(3, 200, num_clusters=32, blur=3)
