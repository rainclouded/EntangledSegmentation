import numpy as np
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt

math_text = """
# Quantum Edge Detection Math
## Phase 1. Imae processing
- Get webcam image
- Resize image
- Convert to greyscale
- Sobel convolution

The sobel convolution uses the following kernels to identify edges
$$
G_x = \\begin{bmatrix}
-1 & 0 & +1 \\\\
-2 & 0 & +2 \\\\
-1 & 0 & +1 
\\end{bmatrix}, \\quad
G_y = \\begin{bmatrix}
-1 & -2 & -1 \\\\
 0 &  0 &  0 \\\\
+1 & +2 & +1 
\\end{bmatrix}
$$
This was used in the classical and threshold versions.

The Threshold version, basically does a second pass and reduces noise (and is arbitrary, need some ML to choose a better one).

It goes pixel by pixel and uses its encoding to really see if it is an edge. 

## Phase 2. Quantum embedding

On the Bloch sphere the |0⟩ location identifies a low probability of an edge while |1⟩ identifies a high probability of an edge.

The quatum entanglement entangles a center qubit representing a 3x3 matrix with the surronding pixels, simulating a convolution.

Each of the urronding qubits are weighted based on their values.

(This was a fun try, but it did not work.)

These were represented as rotations using PauliY gates see the neat demo below.
"""
# Bloch demo component
def bloch_demo(theta):
    # RY(θ) state vector
    alpha = np.cos(theta/2)  # |0> coeff
    beta = 1j * np.sin(theta/2)  # |1> coeff (phase for Y-rotation)
    
    # Bloch vector <X,Y,Z>
    r_x = 2 * np.real(np.conj(alpha) * beta)
    r_y = 2 * np.imag(np.conj(alpha) * beta)
    r_z = np.abs(alpha)**2 - np.abs(beta)**2
    
    fig = plot_bloch_vector([r_x, r_y, r_z], 
                           title=f'RY(θ={theta:.2f})',
                           figsize=(4,4))
    plt.close(fig)  # Close figure to save memory
    return fig
