from PIL import Image

import numpy as np
import gradio as gr
from demo_resources import math_text, bloch_demo
from pipelines import receive_image, receive_image_classical, recieve_kernel


with gr.Blocks() as image_viewer:
    with gr.Tabs():
        with gr.TabItem("Quantum Math"):
            gr.Markdown(math_text)

            with gr.Row():
                theta_slider = gr.Slider(-np.pi,
                                         np.pi, value=0, label="RY(θ)")
            with gr.Row():
                bloch_output = gr.Plot(label="Bloch Sphere")
            theta_slider.change(bloch_demo, theta_slider, bloch_output)

        with gr.TabItem("Live Demo"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(
                        label="Camera", sources="webcam", type="pil")
                    # I have removed the live video stream, it was not working right
                with gr.Column():
                    output_kernel_detected = gr.Image(
                            label="Quantum Kernel")

            with gr.Row():
                    with gr.Column():
                        classical_edge = gr.Image(label="Classical Detector")
                    with gr.Column():
                        output_edge_detected = gr.Image(
                            label="Quantum Threshold")
            with gr.Row():
                with gr.Column():
                    take_screenshot = gr.Button("Classical", variant="primary")
            take_screenshot.click(receive_image_classical,
                                  input_img,  classical_edge)

            with gr.Row():
                with gr.Column():
                    take_screenshot_q1 = gr.Button("Quantum Threshold", variant="primary")
            take_screenshot_q1.click(recieve_kernel, input_img, [
                                  output_edge_detected])
            with gr.Column():
                    take_screenshot_q2 = gr.Button("Quantum Kernel", variant="primary")
            take_screenshot_q2.click(receive_image, input_img, [
                                  output_kernel_detected])

if __name__ == "__main__":

    image_viewer.launch()
