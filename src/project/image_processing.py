from PIL import Image, ImageOps
import numpy as np
from quantum_processing import quantum_kernel
def resize_image(image_name, side_length):
    """Resize/crop image to square dimensions using PIL ImageOps.cover.

    Args:
        image_name: Input PIL Image or numpy array
        side_length: Target square size (width=height)

    Returns:
        PIL Image: Cropped and resized to (side_length, side_length)
    """
    if isinstance(image_name, np.ndarray):
        image_name = Image.fromarray(image_name.astype(np.uint8))
     # We cant have too large an image for now
    resized = ImageOps.cover(image_name, (side_length, side_length))
    return resized


def convert_greybits(image: Image.Image):
    """Convert RGB image to grayscale using what internet says
       are TV-standard luminance weights.

    Args:
        image: PIL RGB image

    Returns:
        np.ndarray: Grayscale image with pixels in range [0, 255]

    Note:
        Y = 0.299R + 0.587G + 0.114B (ITU-R BT.601) the tv standard
    """
    grey_image = np.array(image)
    grey_image = (0.299 * grey_image[:, :, 0]
                  + 0.587 * grey_image[:, :, 1]
                  + 0.114 * grey_image[:, :, 2])  # TV standard for greys (RGB)
    return grey_image


def sobel_convolve(grey_image: np.array, side_length):
    """Manual Sobel convolution with zero-padding boundary conditions.

    Computes horizontal and vertical gradients using 3x3 kernels.

    Args:
        grey_image: Grayscale input (side_length, side_length) [0, 255]
        side_length: Image dimension

    Returns:
        tuple: (x_convolved, y_convolved, grey_normalized)
            - x_convolved: Horizontal gradients [-1, 1]
            - y_convolved: Vertical gradients [-1, 1]  
            - grey_normalized: Intensity [-1, 1]
    """
    # Easy to implement convolution for image processing.
    # These arrays are standard and so currently hard-coded
    x_convolved = np.zeros((side_length, side_length))
    y_convolved = np.zeros((side_length, side_length))
    grey_normalized = np.zeros((side_length, side_length))

    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    gy = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]])

    # add a '0' border to assist with convolution
    padded = np.pad(grey_image, 1, mode='symmetric')

    for x in range(side_length):
        for y in range(side_length):
            convolve_patch = padded[x:x+3, y:y+3]
            x_convolved[x][y] = np.sum(convolve_patch * gx)
            y_convolved[x][y] = np.sum(convolve_patch * gy)
            grey_normalized[x][y] = 2 * (grey_image[x][y] / 255) - 1

    x_max = np.max(np.abs(x_convolved))
    y_max = np.max(np.abs(y_convolved))
    x_convolved = 2 * (x_convolved / x_max) - 1 if x_max > 0 else x_convolved
    y_convolved = 2 * (y_convolved / y_max) - 1 if y_max > 0 else y_convolved

    return (x_convolved, y_convolved, grey_normalized)

def quantum_sobel(grey_image, side_length):
    convolved = np.zeros((side_length, side_length))
    padded = np.pad(grey_image / 255.0, 1, mode='symmetric')
    
    for x in range(side_length):
        for y in range(side_length):
            convolve_patch = padded[x:x+3, y:y+3]
            convolved[x, y] = quantum_kernel(convolve_patch)
    
    return convolved