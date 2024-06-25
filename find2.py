import numpy as np
import sys
import os
from PIL import Image
import shutil

label_path = sys.argv[1]
image_path = sys.argv[2]

label_list = os.listdir(label_path)

new_label = label_path.split('_')[0] + '/'
if not os.path.isdir(new_label):
    os.mkdir(new_label)

new_image = image_path.split('_')[0] + '/'
if not os.path.isdir(new_image):
	os.mkdir(new_image)
# print(new_label)
# print(new_image)
for each_label in label_list:
    each_label_path = os.path.join(label_path, each_label)
    # print(each_label)
    np_label = np.array(Image.open(each_label_path))
    if 2 in np.unique(np_label) or 1 in np.unique(np_label):
        print(f'find 2 in {each_label}')
        # print(image_path)
        each_image_path = os.path.join(image_path, each_label)
        each_image_path = each_image_path.split('.')[0] + '.tif'
        print(f'old_label:{each_label_path} to new_label:{new_label}')
        print(f'old_image:{each_image_path} to new_label{new_image}')
        # os.system("cp -r {0} {1}".format(each_label, new_label))
        # os.system("cp -r {0} {1}".format(each_image, new_image))
        shutil.copy(each_label_path, new_label)
        shutil.copy(each_image_path, new_image)

    
