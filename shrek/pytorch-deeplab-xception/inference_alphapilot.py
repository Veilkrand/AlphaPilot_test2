import os
import numpy as np
from PIL import Image

# PyTorch includes
import torch
import torch.nn as nn
from torchvision import transforms

# Custom includes
from networks import deeplab_xception


class inferenceAlphaPilot():

    def __init__(self,
                 checkpoint_path='checkpoint/checkpoint.pth',
                 imsize=512
                 ):
        super().__init__()

        self.conf_threshold = 0.6 # Any pixel below this confidence value will not be considered as a detection.
        self.checkpoint_path = checkpoint_path

        self.transform = transforms.Compose([
                                             transforms.Resize((imsize,imsize)),
                                             transforms.ToTensor()
                                            ])

        # Create Model
        self.numInputChannels = 3
        self.num_of_classes = 2
        self.output_stride = 8 # Chosen while training the model. Doesn't make diff in inference. Possible Values: 8, 16
        self.net = deeplab_xception.DeepLabv3_plus(nInputChannels=self.numInputChannels,
                                              n_classes=self.num_of_classes,
                                              os=self.output_stride,
                                              pretrained=False)
        self.net = nn.DataParallel(self.net)
        self.net.load_state_dict(torch.load(self.checkpoint_path))

        # Send model to Device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.net.eval()


    def inferenceOnNumpy(self, img):

        # Apply transforms to input numpy image
        img = Image.fromarray(img)
        inputs = self.transform(img)
        inputs = inputs.unsqueeze(0).to(self.device)

        # Forward Pass
        with torch.no_grad():
            outputs = self.net.forward(inputs)

        # Apply a min confidence threshold to prediction
        outputs[outputs < self.conf_threshold] = 0
        predictions = torch.max(outputs, 1)[1]

        # Convert to numpy binary mask
        mask_out = predictions.squeeze(0).cpu().numpy()
        mask_out = mask_out.astype(np.uint8) * 255 # Since only 2 classes 0,1 present, can directly convert to mask

        return mask_out
