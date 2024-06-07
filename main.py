import cv2
import os

from pixelator import Pixelator

input_dir = 'input_folder'
output_dir = 'output_folder'

pix = Pixelator(input_dir, output_dir)

'''
pixel_size is how many real pixels should make up the large pixel
upscale_percentage is how much you want new image to be increased or decreased in size by, default to 100
num_clusters is how many unique colours you want in the image, increasing it increases processing time
blur applies a blur filter on the original image before doing the pixelation. This helps with smoothing. It has to be at least 1.
You might also get an error if the blur is an even number
'''
pix.kmeans(pixel_size=9, upscale_percentage=100, num_clusters=64, blur=3)
