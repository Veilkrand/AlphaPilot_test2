#Class for image augumentation
import cv2
import pandas as pd
import numpy as np
class ImgAugumentation:
    #def __init__(self, image,scale_range,kernel_size):
        #self.image=image
        #self.scale_range=scale_range
        #self.kernel_size=kernel_size
        
    def brightness_images(self,image):
        post_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        post_img[:,:,2] = np.multiply(post_img[:,:,2],random_bright)
        post_img = cv2.cvtColor(post_img,cv2.COLOR_HSV2RGB)
        return post_img
    # Crop away unwanted regions from the orginal image  
    def crop_img(self, image, mask):
        shape = image.shape
        image = image[0:shape[0]-20,0:shape[1]]
        image = resize_img(image, 64, 64)
        mask = resize_img(mask, 64, 64)
        return image, mask
    def gaussian_noise(self, image, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Flip images to improve training
    def flip_image_horz(self,image,mask):
        flip_image = image.copy()
        #num = np.random.randint(2)
        #if num == 0:
        flip_image = cv2.flip(image, 1)
        flip_mask = cv2.flip(mask, 1)
        return flip_image, flip_mask
    def flip_image_ver(self,image,mask):
        flip_image = image.copy()
        #num = np.random.randint(2)
        #if num == 0:
        flip_image = cv2.flip(image, -1)
        flip_mask = cv2.flip(mask, -1)
        return flip_image, flip_mask
    
    # Use this function after NearMap has provided chessboard calibiration images
    # Implement calibration on the images that will be used
    def undistort(self, img, read=True, display=True, write=False):
        if read:
            img = plt.imread(img)
        img_size = (img.shape[1], img.shape[0])
    #img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
    #dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        if write:
            cv2.imwrite('Undistorted/test6.jpg',dst)
    # Visualize undistortion
        if display:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            img_RGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax1.imshow(img_RGB)
            ax1.set_title('Original Image', fontsize=30)
            dst_RGB=cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            ax2.imshow(dst_RGB)
            ax2.set_title('Undistorted Image', fontsize=30)
        else:
            return dst
    #Create a binary and border masks from a polygon of pixel coordinates in an array list
    #pixel_coords
    
    def create_binary_mask(self, pixel_coords, input_image):
        matrix = np.zeros_like(input_image[:,:,0],dtype=np.uint8)
        #print('input image size',np.shape(matrix))
        if len(pixel_coords) > 0: 
            for polygon in pixel_coords:
                polygon = polygon.reshape((-1,1,2))
                border_img = cv2.polylines(matrix, [polygon], True, (1,2,0),3)
                binary_img = cv2.fillPoly(matrix, [polygon], 255)
        else:
            binary_img = matrix
            border_img = matrix
        return binary_img, border_img
    
    def trans_image(self, image, mask, trans_range):
        # Translation augmentation
         

        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2

        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        rows,cols,channels = image.shape
 

        image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
        mask_tr = cv2.warpAffine(mask,Trans_M,(cols,rows))
        return image_tr,mask_tr 
 
  
    def stretch_image(self, image, mask, scale_range):
    # Stretching augmentation
        tr_x1 = scale_range*np.random.uniform()
        tr_y1 = scale_range*np.random.uniform()
        p1 = (tr_x1,tr_y1)
        tr_x2 = scale_range*np.random.uniform()
        tr_y2 = scale_range*np.random.uniform()
        p2 = (image.shape[1]-tr_x2,tr_y1)

        p3 = (image.shape[1]-tr_x2,image.shape[0]-tr_y2)
        p4 = (tr_x1,image.shape[0]-tr_y2)

        pts1 = np.float32([[p1[0],p1[1]],
                       [p2[0],p2[1]],
                       [p3[0],p3[1]],
                       [p4[0],p4[1]]])
        pts2 = np.float32([[0,0],
                       [image.shape[1],0],
                       [image.shape[1],image.shape[0]],
                       [0,image.shape[0]] ]
                       )

        M = cv2.getPerspectiveTransform(pts1,pts2)
        image = cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))
        image = np.array(image,dtype=np.uint8)
        mask = cv2.warpPerspective(mask,M,(mask.shape[1],mask.shape[0]))
        mask = np.array(mask,dtype=np.uint8)
        return image, mask 
    
    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])
    
    def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                        np.ones_like(image_array))
        alpha = 2 * np.random.random() * self.contrast_var
        alpha = alpha + 1 - self.contrast_var
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)
    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array + (1 - alpha) * gray_scale[:, :, None]
        return np.clip(image_array, 0, 255)
    
    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1':image_array},
                {'predictions':targets}]
