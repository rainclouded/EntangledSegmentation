from PIL import Image, ImageOps
import numpy as np
import pennylane as qml
side_length = 512
dev = qml.device('default.qubit', wires = 3)


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

@qml.qnode(device = dev)
def test_pixel(x_angle, y_angle, grey_angle):
    # This is how to determne if its foreground or background

    # encode in qubit here:
    
    qml.RY(x_angle, wires = 0)
    qml.RY(y_angle, wires = 1)
    qml.RY(grey_angle, wires = 2)

    coeffs = [1,1,1]
    obs = [qml.RX(0), qml.RY(1), qml.RZ(2)]
    H = qml.Hamiltonian(coeffs, obs)
    return qml.expval(H)


def test_full_image(x_convolved, y_convolved, grey_normalized):
    edge_image = np.zeroes(side_length, side_length)
    for x in range(side_length):
        for y in range(side_length):
            x_angle = x_convolved[x][y]
            y_angle = y_convolved[x][y]
            grey_angle = grey_normalized[x][y]
            edge_image[x][y] = 1 if test_pixel(x_angle, y_angle, grey_angle) > 0.5 else 0
    return edge_image

def test_image():
    resized = None
    with Image.open('./test_files/test_image.png') as img:
        resized = resize_image(img)
    grey_image = convert_greybits(resized)
    x_conv, y_conv, grey_norm = sobel_convolve(grey_image)
    final_edge_detection = test_full_image(x_conv, y_conv, grey_norm)
    Image.fromarray((final_edge_detection*255).astype(np.uint8), 'L').save('quantum_edges.png')
