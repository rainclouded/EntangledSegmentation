from PIL import Image, ImageOps
import numpy as np
side_length = 512

def resize_image(image_name):
     # We cant have too large an image for now
    resized  = ImageOps.cover(image_name, side_length, (0.5, 0.5))
    return resized

def convert_greybits(image:Image.Image):
    grey_image = np.zeroes(side_length,side_length)
    for x in range(side_length):
        for y in range(side_length):
            R,G,B = image[x,y]
            grey_image[x,y] = 0.299 * R + 0.587 * G + 0.114*B # TV standard for greys

def sobel_convolve(grey_image:np.array):
    # Easy to implement convolution for image processing.
    # These arrays are standard and so currently hard-coded
    x_convolved = np.zeros(shape=grey_image)
    y_convolved = np.zeros(shape=grey_image)
    grey_normalized = np.zeros(shape = grey_image)

    gx = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])

    gy = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]])
    
    padded = np.pad(grey_image, 1) # add a one value border to assist with convolution

    for x in range(side_length):
        for y in range(side_length):
            convolve_patch = padded[x:x+3, y:y+3]
            x_conv = np.sum(convolve_patch * gx)
            y_conv = np.sum(convolve_patch * gy)
            x_convolved[x][y] = 2 * (x_conv / max(abs(x_conv))) - 1
            y_convolved[x][y] = 2 * (y_conv / max(abs(y_conv))) - 1
            grey_normalized[x][y] = 2 * (grey_image / 255) - 1
    return (x_convolved, y_convolved, grey_normalized)
            
def collapse_edges(convolved:np.array):
    edge_image = np.zeroes(side_length,side_length)
    threshold = .5 # to be updated, but what should be edge or background
    for x in range(side_length):
        for y in range(side_length):
            edge_image[x][y] = 0 if convolved[x][y] > threshold else 1
    return edge_image

def int_to_greycode(value):
    # use grey code because values differ by a single but flip
    # since this will be simulated, not an actual requirement...
    return np.unpackbits(value ^ (value << 1)) # this is a neat hack!
    

def convert_to_angle(edge_image:np.array):
    angles = edge_image.flatten() * np.pi # we want to start with either Pi or 0
    # This will be for Pauli Z rotation

def get_pixel(row,col, pixel_angles):
    selected_angle = int_to_greycode(row) + int_to_greycode(col) + [pixel_angles[row * side_length + col]]

