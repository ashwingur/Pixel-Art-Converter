import cv2
import os

from pixelator import Pixelator

input_dir = 'input_folder'
output_dir = 'output_folder'

pix = Pixelator(input_dir, output_dir)

pix.kmeans(pixel_size=4, upscale_percentage=100, num_clusters=64, blur=5)
