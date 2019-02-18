from matplotlib import pyplot as plt
import imutils
import numpy as np
import cv2

class Corners:

    def __init__(self):

        # Params
        self._epsilon_constant = 0.1  # Bigger simpler shape fitting # 0.04
        self._min_pixels_size = 10  # Min size of shapes in min rect area


    def process_image_path(self, path):

        img_original = cv2.imread(path)

        img_prep = self.prepare_image(img_original)

        img_shapes, shapes = self.find_quads(img_prep)

        inner_shape = self.get_inner_area_corners_from_results(shapes)

        img_result = self.draw_points_array(img_original, inner_shape)

        return img_result, inner_shape

    def draw_points_array(self,image, points):
        for i in range(0, len(points)):
            cv2.circle(image, (int(points[i, 0]), int(points[i, 1])), 10, (255, 0, 0), 7)
        return image

    def get_inner_area_corners_from_results(self, results):
        assert len(results) == 2, 'I got more than 2 shapes for a gate'

        inner_shape = None

        if results[0]['area'] > results[1]['area']:
            inner_shape = results[1]
        else:
            inner_shape = results[0]

        assert len(inner_shape['corners']) == 5, 'I didn\'t all 5 points of the shape'

        corners = inner_shape['corners'][1:] # Skip centroid as first element of the points
        return corners


    def prepare_image(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th1 = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)[1]
        return th1

    def find_quads(self, img):

        # Params
        _epsilon_constant = self._epsilon_constant
        _min_pixels_size = self._min_pixels_size
        # ---

        img_result = img.copy()

        cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        results = []

        for c in cnts:

            # Perform shape approximation
            epsilon = _epsilon_constant * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)

            # 4 sides?
            if (len(approx) == 4):

                cv2.drawContours(img_result, [c], -1, (0, 255, 0), 5)

                (x, y, w, h) = cv2.boundingRect(approx)
                if w >= _min_pixels_size and h >= _min_pixels_size:

                    result = {}

                    #print(x, y, w, h, len(c))

                    area = w * h
                    result['area'] = area

                    M = cv2.moments(c)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    result['centroid'] = (cx, cy)

                    # create mask for edge detection
                    gray = np.float32(img)
                    mask = np.zeros(gray.shape, dtype="uint8")
                    cv2.fillPoly(mask, [approx], (255, 255, 255))

                    dst = cv2.cornerHarris(mask, 5, 3, 0.04)
                    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
                    dst = np.uint8(dst)
                    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

                    result['corners'] = corners

                    results.append(result)

        return img_result, results

