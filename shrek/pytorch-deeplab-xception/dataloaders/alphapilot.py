#!/usr/bin/python3

from __future__ import division, print_function

import fnmatch
import os

import numpy as np
import torch
from PIL import Image
from torch import FloatTensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import cv2
import imgaug as ia
from imgaug import augmenters as iaa


class AlphaPilotSegmentation(Dataset):
    """
    Dataset class for satellite images. Contains variable number of classes. Data augmentation with imgaug.
    """
    INPUT_IMG_EXTENSIONS = ['.jpg', '.jpeg', '.JPG']
    LABEL_IMG_EXTENSIONS = ['.png']


    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)


    @staticmethod
    def _load_input_image(path):
        _img = Image.open(path).convert('RGB')
        return np.array(_img)


    @staticmethod
    def _load_label_image(path):
        _img = Image.open(path).convert('L')
        return np.array(_img)[..., np.newaxis]


    def __init__(self,
                 input_dir='data/dataset/train/images',
                 label_dir='data/dataset/train/labels',
                 transform=None,
                 input_only=None,
                 ):
        """
        Args:
            input_dir (str): Path to folder containing the input images.
                Expects images in them to be in format as in list INPUT_IMG_EXTENSIONS.
            label_dir (str): Path to folder containing the labels.
                Expects images in them to be in format as in list LABEL_IMG_EXTENSIONS.
                If running inference on test set without labels, pass label_dir as None
            transform (imgaug transforms): Transforms to be applied to the imgs
            input_only (list, str): List of transforms that are to be applied only to the input img
        """
        super().__init__()

        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.input_only = input_only

        assert os.path.isdir(input_dir), 'This directory does not exist: %s' % (input_dir)
        self.datalist_input = sorted((os.path.join(self.input_dir, img)) for img in os.listdir(self.input_dir)
                                     if self._isimage(img, self.INPUT_IMG_EXTENSIONS))

        if self.label_dir:
            assert os.path.isdir(label_dir), 'This directory does not exist: %s' % (label_dir)
            self.datalist_label = sorted((os.path.join(self.label_dir, img)) for img in os.listdir(self.label_dir)
                                        if self._isimage(img, self.LABEL_IMG_EXTENSIONS))
            assert(len(self.datalist_input) == len(self.datalist_label)
                ), 'Number of images and Number of labels should be the same. Found {} images, {} labels'.format(
                    len(self.datalist_input), len(self.datalist_label))


    def __len__(self):
        return len(self.datalist_input)


    def __getitem__(self, index):
        _img = self._load_input_image(self.datalist_input[index])
        if self.label_dir:
            _label = self._load_label_image(self.datalist_label[index])


        #TODO: Convert label to int at this point, same dtype as _img. See if that helps with the interpolation problem.
        # Apply processing and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            if self.label_dir:
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        _img = np.ascontiguousarray(_img) # To prevent errors from negative stride, as caused by fliplr()
        _img_tensor = transforms.ToTensor()(_img)
        if self.label_dir:
            _label_tensor = transforms.ToTensor()(_label.astype(np.float32)) # Without conversion of numpy to float, the numbers get normalized
        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        # return _img_tensor, _label_tensor, os.path.splitext(os.path.basename(self.datalist_input[index]))[0]
        return _img_tensor, _label_tensor

    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

    augs = None  # augs_train, augs_test, None
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = AlphaPilotSegmentation(
        input_dir='data/dataset/train/images',
        label_dir='data/dataset/train/labels',
        transform=augs,
        input_only=input_only
    )

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label, filename = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)

        break