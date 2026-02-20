from PIL import Image, ImageOps
import numpy as np
side_length = 512

def resize_image(image_name):
     # We cant have too large an image for now
    resized  = ImageOps.cover(image_name, side_length, (0.5, 0.5))
    return resized

def convert_greybits(image:Image.Image):
    grey_image = np.zeroes(side_length,side_length)
    for x in side_length:
        for y in side_length:
            R,G,B = image[x,y]
            grey_image[x,y] = 0.299 * R + 0.587 * G + 0.114*B

