# yolo-detector

This Python project allows you to perform YOLOv3 object detection without using the Darknet library. It leverages the OpenCV DNN module to load YOLOv3 parameters from a JSON file, making it a simpler solution for those who have trained models in Darknet and want to use them in Python.

## Key Features
* Object Detection: Detect objects in images using YOLOv3.
* Save Detections: Save images with bounding box detections.
* Batch Processing: Easily process multiple images in a batch.
* Video Support: Extend the project to handle YOLOv3 detection in MP4 videos.

## Requirements
Make sure you have the following Python libraries installed:

* Numpy
* OpenCV
* Time
* JSON

## Getting Started
1. Clone this repository to your local machine.
2. Install the requiments libraries using 'pip'.
3. Add the yolo files to the YoloWeights folder.
4. Provide the json file with your model file names.
5. Run the script and enjoy the simplified YOLO detection in Python!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
This project simplifies YOLOv3 object detection for Python users.
Special thanks to the OpenCV and YOLO communities for their contributions.