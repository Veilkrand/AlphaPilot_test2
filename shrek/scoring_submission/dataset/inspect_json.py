import json
import glob
import os

# Read test images
# test_imgs_path = '/home/shrek/alphapilot/AlphaPilot_test2/solution/input_images/*.JPG'
# filenames_img_full = glob.glob(test_imgs_path)

# filenames_img = []
# for filename in filenames_img_full:
#     filename = os.path.basename(filename)
#     filenames_img.append(filename)

# filenames_img = sorted(filenames_img)
# # print(filenames_img)
# for name in filenames_img:
#     print(name)
# print('\n\n' + '-' * 50 + '\n\n')

# print('len_json: ', len(filenames))
# print('len_imgs: ', len(filenames_img))


# open json
pred_dict = {}
with open('testing_leaderboard_GT_labels_v3_XY_format.json', 'r') as f:
    pred_dict = json.load(f)

# make list of filenames in json
filenames = []
for img in pred_dict:
    if not "outer_polygon" in img["Label"]:
        filenames.append(img["External ID"])

filenames = sorted(filenames)
# print(filenames)
for name in filenames:
    print(name)
print('\n\n' + '-' * 50 + '\n\n')

