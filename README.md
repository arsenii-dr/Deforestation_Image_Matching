# README: Image Matching for Deforestation Detection Using Sentinel-2 and LoFTR Model #

## Introduction ##

This project focuses on _extracting keypoints and matching_ `Sentinel-2` images. By leveraging the `LoFTR` (Local Feature TRansformer) model, the tool provides an efficient and scalable solution for comparing high-resolution satellite images to identify changes in forest cover over time.

## Features ##

- __Deforestation Detection:__ Matches key points between multi-temporal Sentinel-2 images.

- __LoFTR Integration:__ Uses a state-of-the-art feature matching model to ensure robust and accurate keypoint detection. Handles challenging scenarios such as lighting changes, cloud cover, and partial occlusions.

## Project structure ##

- __/data__
    - __/input:__ Raw data from kaggle. _/deforestation_in_ukraine_
    - __/prepared_images:__ Extracted and selected images.
- __/notebooks:__ Jupyter notebook, where data and LoFTR model were explored.
- __/results:__ Matched images vizualization.
- __LoFTR.py:__ Defining and fine-tuning pretrained model. 
- __match.py:__ Matching 4 selected images and visualization.
- __README.md:__ Project description
- __requirements.txt:__ Dependencies for the project

## How to use ##

1. Install `requirements.txt`.
```
pip install -r requirements.txt
```
2. Run Image Matching:

Use the `matcher.py` script to match images and identify keypoints.

