![Entangled Segmentation](image.png)
# Entangled Segmentation Hackathon Submission

This project was meant to mix computer vision and quantum computing.

@[rainclouded](https://github.com/rainclouded) is the only team member.


# Run this

```bash
docker compose build  && docker compose up
```

To run the project. Navigate to http://127.0.0.1:7860/


Allow the webcam access to the webpage

then press the camera button to take a snapshot (without this you will get an error).

Then press the corresponding buttons to run the edge detections.

Classical is OpenCV sobel, Quantum threshold does sobel then basically does a second pass for greater edge detection, the third quantum kernel is an attempt to embed a sobel kernel in the qubits - it didn't work out (but was worth trying). Update - is working to an extent now.
