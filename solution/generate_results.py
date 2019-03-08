# Core imports
import numpy as np
from scipy.spatial import distance as dist
import cv2

# Inference imports
from inference_alphapilot import inferenceAlphaPilot

# Estimator imports
from QuadEstimator import QuadEstimator


class GenerateFinalDetections():


    def __init__(self):


        print('loading modules...')
        # Load estimator
        self.estimator = QuadEstimator()


        model_checkpoint = 'checkpoints/checkpoint.pth'
        # Load inference
        # NOTE: set map_location='cpu' if CUDA is not available
        self.inference = inferenceAlphaPilot(checkpoint_path=model_checkpoint,
                                        conf_threshold=0.97,
                                        imsize=512)
        print('Ready!')

 
    def order_points(self, pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
     
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
     
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
     
        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
     
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")

    def predict(self, img_original):
        
        h, w = img_original.shape[:2]

        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        

        print('--- Runing inference ---')
        img_mask = self.inference.inferenceOnNumpy(img_bgr)
        
        #img_resized_mask = cv2.resize(img_mask, (1296, 864)) 
        img_resized_mask = cv2.resize(img_mask, (w, h))
        
        corners, img_corners = self.estimator.process_img(img_resized_mask, gray=True)
        
        if corners is not None:
            ordered_corners = self.order_points(corners)
            poly = ordered_corners.flatten().tolist()

            poly.append(0.5)

            result = []
            result.append(poly)
        else:
            result = [[]]

        print('Solution', result)
        
        return result