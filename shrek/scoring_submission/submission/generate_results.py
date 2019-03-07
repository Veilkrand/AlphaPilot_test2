# This script is to be filled by the team members.
# Import necessary libraries
# Load libraries
import json
import cv2
import numpy as np

# Implement a function that takes an image as an input, performs any preprocessing steps and outputs a list of bounding box detections and assosciated confidence score.
from QuadEstimator import QuadEstimator
from inference_alphapilot import inferenceAlphaPilot

class GenerateFinalDetections():
    def __init__(self):
        self.estimator = QuadEstimator()
        self.inference = inferenceAlphaPilot(checkpoint_path='checkpoint/checkpoint.pth',
                                             conf_threshold=0.97,
                                             imsize=512)

    def predict(self, img):
        print('img', img.shape)
        # Run inference on model
        mask = self.inference.inferenceOnNumpy(img)

        # Get corners
        mask_orig_size = cv2.resize(mask, (img.shape[1], img.shape[0]))
        print('mask_orig_size', mask_orig_size.shape)
        poly, img_result = self.estimator.process_img(mask, gray=True)

        # Convert numpy result to list
        poly = np.reshape(poly, (1, 8))
        list_coords = poly.astype(np.uint32).tolist()
        return list_coords
