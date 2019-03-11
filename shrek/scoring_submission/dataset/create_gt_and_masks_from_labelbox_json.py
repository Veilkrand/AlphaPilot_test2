import json
import os
import imageio
import numpy as np
from PIL import Image, ImageDraw

# testing annotation all together
def get_mask(img_shape, poly):

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)

    draw.polygon(xy=poly, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)

    return mask


path_labelbox_json = 'testing_leaderboard_GT_labels_v3_XY_format.json'
path_output_gt_json = 'testing_leaderboard_GT_labels_v3_alphapilot_format.json'
path_output_masks = 'Data_Test_Leaderboard_masks/'
img_shape = (864, 1296)
gt_json = {}

# open json
input_dict = {}
with open(path_labelbox_json, 'r') as f:
    input_dict = json.load(f)


for img_dict in input_dict:
    print('creating mask for', img_dict["External ID"])

    inner_poly = []
    outer_poly = []

    # build list of inner polygon
    for pair in img_dict["Label"]["inner_polygon"][0]["geometry"]:
        inner_poly.append(pair["x"])
        inner_poly.append(pair["y"])

    # Build list of outer polygon
    for pair in img_dict["Label"]["outer_polygon"][0]["geometry"]:
        outer_poly.append(pair["x"])
        outer_poly.append(pair["y"])

    # Add this to the json of organizer's format to be exported later
    gt_json.update( {img_dict["External ID"]: [inner_poly]} )

    # Create mask of outer poly
    outer_mask = get_mask(img_shape, outer_poly)

    # Create mask of inner poly
    inner_mask = get_mask(img_shape, inner_poly)

    # Subtract to get mask of gate
    gate_mask = outer_mask ^ inner_mask

    # Output mask of gate
    output_gate_label = np.zeros(img_shape, dtype=np.uint8)
    output_gate_label[gate_mask] = 1
    output_filename = os.path.splitext(img_dict["External ID"])[0] + '.png'
    imageio.imsave(os.path.join(path_output_masks, output_filename), output_gate_label)

    # Write output GT json in alphapilot format
    with open(path_output_gt_json, 'w') as f:
        json.dump(gt_json, f)
