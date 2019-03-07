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

        mask = self.inference.inferenceOnNumpy(img)
        poly, img_result = self.estimator.process_img(mask, gray=True)


        print('img_result:', img_result.shape)
        print('poly:', len(poly))

        return poly

        # np.random.seed(self.seed)
        # n_boxes = np.random.randint(4)
        # if n_boxes>0:
        #     bb_all = 400*np.random.uniform(size = (n_boxes,9))
        #     bb_all[:,-1] = 0.5
        # else:
        #     bb_all = []
        # return bb_all.tolist()

