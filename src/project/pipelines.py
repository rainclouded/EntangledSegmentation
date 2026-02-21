import numpy as np
from PIL import Image
import cv2
from image_processing import resize_image, convert_greybits, sobel_convolve, quantum_sobel
from quantum_processing import test_pixel

# The quantum algorithm is quite slow...
# So it will get downsampled
quantum_side_length = 128
classical_side_length = 512


def test_full_image(x_convolved, y_convolved, grey_normalized, side_length):
    """Vectorized quantum edge detection over entire image.

    Args:
        x_convolved: Horizontal Sobel gradients (side_length, side_length)
        y_convolved: Vertical Sobel gradients (side_length, side_length)
        grey_normalized: Normalized intensities (side_length, side_length)
        side_length: Image dimension

    Returns:
        np.ndarray: Binary edge map (side_length, side_length) {0, 1}
    """
    edge_image = np.zeros((side_length, side_length))
    x_angles = x_convolved.flatten()
    y_angles = y_convolved.flatten()
    greys = grey_normalized.flatten()

    results = test_pixel(x_angles, y_angles, greys)

    edge_image = (results > 0.173).astype(int).reshape(grey_normalized.shape)
    return edge_image


def process_quantum_threshold(grey_image):
    """quantum edge detection pipeline.

    Args:
        grey_image: Grayscale input (quantum_side_length, quantum_side_length)

    Returns:
        np.ndarray: Edge map [0, 255] uint8
    """
    x_conv, y_conv, grey_norm = sobel_convolve(grey_image, quantum_side_length)
    final_edge_detection = test_full_image(
        x_conv, y_conv, grey_norm, quantum_side_length)
    return (final_edge_detection*255).astype(np.uint8)


def process_quantum(grey_image):
    edge_map = quantum_sobel(grey_image, quantum_side_length)
    return ((edge_map < 0) * 255).astype(np.uint8)

def process_classical(resized):
    """Classical Sobel edge detection using  OpenCV.

    Args:
        resized: RGB input (classical_side_length, classical_side_length, 3)

    Returns:
        np.ndarray: Edge map [0, 255] uint8
    """
    img_open_cv_grey = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_open_cv_grey, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_open_cv_grey, cv2.CV_64F, 0, 1, ksize=3)

    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return edges


def receive_image(img):
    """Get the image for quantum detection. webcam → quantum edges.

    Args:
        img: PIL Image from Gradio webcam

    Returns:
        np.ndarray: Quantum edge-detected image
    """
    resized = resize_image(img, quantum_side_length)
    grey_image = convert_greybits(resized)
    return process_quantum(grey_image)

def recieve_kernel(img):
    resized = resize_image(img, quantum_side_length)
    grey_image = convert_greybits(resized)
    return process_quantum_threshold(grey_image)


def receive_image_classical(img):
    """Get the image for classical detection. webcam → classical edges.

    Args:
        img: PIL Image from Gradio webcam 

    Returns:
        np.ndarray: OpenCV edge-detected image  
    """
    resized = np.array(resize_image(img, classical_side_length))
    return process_classical(resized)


def test_image():  # Literally just for testing, can be deleted
    resized = None
    with Image.open('./test_files/test_image.png') as img:
        resized = resize_image(img, quantum_side_length)
    print('resized')
    grey_image = convert_greybits(resized)
    print('greyed')
    x_conv, y_conv, grey_norm = sobel_convolve(grey_image, quantum_side_length)
    print('sobel')
    final_edge_detection = test_full_image(
        x_conv, y_conv, grey_norm, quantum_side_length)
    print('final')
    Image.fromarray((final_edge_detection*255).astype(np.uint8),
                    'L').save('quantum_edges.png')
