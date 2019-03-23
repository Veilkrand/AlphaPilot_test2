import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import argparse
from collections import OrderedDict
from imgaug import augmenters as iaa
import imgaug as ia
import imageio

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn as nn

# Custom includes
from dataloaders.alphapilot import AlphaPilotSegmentation
from dataloaders import utils
from networks import deeplab_xception

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_images_path", required=True, help="Path to test dataset - input images")
parser.add_argument("-l", "--label_images_path", required=True, help="Path to test dataset - labels")
parser.add_argument("-c", "--checkpoint_path", required=True, help="Path to a checkpoint file to be loaded")
parser.add_argument("-r", "--result_folder", default="", help="prefix to be added to name of file, saved to data/results")
parser.add_argument("-s", "--imsize", type=int, default=512, help="Size to resize the imgs to before feeding into network")
parser.add_argument("-t", "--conf_threshold", type=float, default=0.6, help="Results below this conf will be discarded")
parser.add_argument("-n", "--num_of_classes", type=int, default=2, help="Num of classes in the checkpoint to be loaded")
parser.add_argument("-o", "--output_stride", type=int, default=8, help="Output stride of deeplab model")
args = parser.parse_args()


testBatchSize = 1  # Testing batch size
conf_threshold = args.conf_threshold # Any pixel below this confidence value will not be considered as a detection.
results_store_dir = 'data/results'
results_store_dir = os.path.join(results_store_dir, args.result_folder)
output_masks_dir = os.path.join(results_store_dir, 'masks')
if not os.path.isdir(output_masks_dir):
    os.makedirs(output_masks_dir)
else:
    raise ValueError('This dir already exists: ', results_store_dir)

def print_iou_per_class(miou_per_class):
    '''Makes a string describing the IoU of each class, with it's name.
    '''
    Binary_class_index = {
        'Background': 0,
        'Gate'      : 1,
    }

    ret_str = '    IoU per class:\n'
    class_index_dict = Binary_class_index

    for key in class_index_dict:
        ret_str = ret_str + ('    ' + key + (22-len(key))*' ' + '%.2f\n'%(miou_per_class[class_index_dict[key]]))

    return ret_str



numInputChannels = 3
net = deeplab_xception.DeepLabv3_plus(nInputChannels=numInputChannels, n_classes=args.num_of_classes, os=args.output_stride, pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Let's use", torch.cuda.device_count(), "GPUs!")

print("Initializing weights from: {}...".format(args.checkpoint_path))
net = nn.DataParallel(net)  #Because models are trained with dataparallel and saved directly, we need to use it too.
net.load_state_dict(torch.load(args.checkpoint_path))
net.to(device)


augs_test = iaa.Sequential([
    iaa.Scale((args.imsize, args.imsize), 0), # Resize the img
])

db_test = AlphaPilotSegmentation(
    input_dir=args.input_images_path, label_dir=args.label_images_path,
    transform=augs_test,
    input_only=None
)
testloader = DataLoader(db_test, batch_size=testBatchSize, shuffle=False, num_workers=4, drop_last=True)
num_img_ts = len(testloader)

#===========================Inference Loop=====================================#
net.eval()
total_iou = 0.0
miou = 0.0
miou_per_class = [0] * args.num_of_classes
num_images_per_class = [0] * args.num_of_classes
for ii, sample_batched in enumerate(testloader):
    inputs, labels, sample_filename = sample_batched
    print('  sample_filename: ', sample_filename[0])

    # Forward pass of the mini-batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = net.forward(inputs)

    # Apply a min confidence threshold
    # outputs[outputs < conf_threshold] = 0
    predictions = torch.max(outputs, 1)[1]

    inputs = inputs.cpu()
    labels = labels.cpu().type(torch.FloatTensor)
    predictions = predictions.cpu().type(torch.FloatTensor)

    if args.label_images_path:
        _total_iou, per_class_iou, per_class_img_count = utils.get_iou(predictions, labels.squeeze(1), n_classes=args.num_of_classes)
        total_iou += _total_iou
        for i in range(len(per_class_iou)):
            miou_per_class[i] += per_class_iou[i]
            num_images_per_class[i] += per_class_img_count[i]

    # Save the model output, 3 imgs in a row: Input, Prediction, Label
    imgs_per_row = 3
    predictions_colormap = utils.decode_seg_map_sequence(predictions.squeeze(1).numpy()).type(torch.FloatTensor)
    labels_colormap = utils.decode_seg_map_sequence(labels.squeeze(1).numpy()).type(torch.FloatTensor)
    sample = torch.cat((inputs, predictions_colormap, labels_colormap), 0)
    img_grid = make_grid(sample, nrow=testBatchSize*imgs_per_row, padding=2)
    save_image(img_grid, os.path.join(results_store_dir, sample_filename[0] + '-results.png'))

    mask_out = predictions.squeeze(0).numpy() * 255
    imageio.imwrite(os.path.join(results_store_dir, 'masks', sample_filename[0] + '.png'), mask_out.astype(np.uint8))

    # Calculate mean IoU per class and overall
    print('  image num : %03d' % (ii * testBatchSize))
    if args.label_images_path:
        if ii % num_img_ts == num_img_ts - 1:
            miou = total_iou / (ii * testBatchSize + inputs.shape[0])
            for i in range(len(miou_per_class)):
                if num_images_per_class[i] == 0:
                    miou_per_class[i] = -1
                else:
                    miou_per_class[i] = miou_per_class[i] / num_images_per_class[i]

print('\n    %s mIoU: %.1f%%\n'%(args.result_folder, miou*100))
print(print_iou_per_class(miou_per_class))

filename = args.result_folder+'-mIoU.txt'
with open(os.path.join(results_store_dir,filename), 'w') as f:
    f.write('\n    %s mIoU: %.1f%%\n'%(args.result_folder, miou*100))
    f.write(print_iou_per_class(miou_per_class))

print('Wrote results into file ', filename)
