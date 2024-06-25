import sys
import numpy as np
from PIL import Image
import os
from skimage import io

root = sys.argv[1]

label = os.path.join(root, 'Labels')

label_path_list = [os.path.join(label, each_label) for each_label in os.listdir(label)]

for each_label in label_path_list:
    label_name = each_label.split('/')[-1]
    image_name = label_name.split('.')[0]
    label = Image.open(each_label)
    label_np = np.array(label)
    label_np_unique = np.unique(label_np)
    print(label_np_unique)

    if 2 in label_np_unique:
        label_np[label_np == 2] = 1

    print(f'save {image_name} : {np.unique(label_np)}')

    io.imsave(each_label, label_np)

