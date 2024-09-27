# PlateVision

## Introduction

The project is targeting to detect vehicles and read the license plate. The project works best on Azerbaijani license plates and it has reach to 98% accuracy on a highway video with fast moving vehicles (unfortunately the test video is not open to public so I cannot share it, but you can test out your own video).

The project is open-source and mainly intended for educational purposes.

The code is **not** on its best version, so you might come up with bugs and/or unwanted results. Please open an issue if that is the case.

## Quick Start

Create environment and install the requirements (the code has been test on Python version 3.12.5):

```conda create --name my_env ``` 

```conda activate my_env```

```pip install -r requirements.txt```

Make the corresponding changes in ```main.py``` file if needed and run following:

```python main.py```

Results will be saved to ```videos/results/``` directory.

For quick testing run the following script to make recognition on test image (you might want to change the path if you want to test your own image):

```python test_single_frame.py```

Results will be saved to ```results/``` directory.


## Acknowledgments / Used Repositories

This project makes use of the following open-source repositories:

- [ultralytics](https://github.com/ultralytics/ultralytics): Yolov8 has been used for making vehicle detection and tracking.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): PaddleOCR has been used to read the license plate on detected vehicles.
- [openvino](https://github.com/openvinotoolkit/openvino): Openvino has been used for optimized inference of YOLO model.
