import pennylane as qml
# kinda like this but bigger https://pennylane.ai/qml/demos/tutorial_quanvolution
# Pennylane setup, use lightning for speed! (its not much fasrer)
dev = qml.device('lightning.qubit', wires=3)
coeffs = [1, 1, 1]
obs = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
H = qml.Hamiltonian(coeffs, obs)

import numpy as np
dev_entanglement = qml.device('lightning.qubit', wires=9)



@qml.qnode(dev_entanglement)
def quantum_kernel(patch, coefficients):
    # Encode each pixel like its a sobel kernel
    
    for i, pixel in enumerate(patch.flatten()):
        qml.RY((pixel * np.pi)*coefficients[i], wires=i) # if its a 1 it rotates by pi
    
    # Entangling the qubits
    # Imitating the sobel convolution
    qml.CNOT(wires=[0,4])
    qml.CNOT(wires=[1,4])
    qml.CNOT(wires=[2,4])
    qml.CNOT(wires=[3,4])
    qml.CNOT(wires=[5,4])
    qml.CNOT(wires=[6,4]) 
    qml.CNOT(wires=[7,4])
    qml.CNOT(wires=[8,4])



    # The center qubit gets measured
    return qml.expval(qml.PauliZ(4))



@qml.qnode(device=dev)
def test_pixel(x_angle, y_angle, grey_angle):
    """Quantum edge detection circuit: encodes Sobel gradients in RY rotations.

    Args:
        x_angle: Horizontal Sobel gradient [-π/2, π/2] → qubit 0
        y_angle: Vertical Sobel gradient [-π/2, π/2] → qubit 1  
        grey_angle: Normalized intensity [-1, 1] → qubit 2

    Returns:
        float: ⟨X₀ + Y₁ + Z₂⟩ ∈ [-3, 3], edge if > 0.5
    """
    # This is how to determne if its foreground or background
    # encode in qubit here:

    qml.RY(x_angle, wires=0)
    qml.RY(y_angle, wires=1)
    qml.RY(grey_angle, wires=2)

    return qml.expval(H)