import numpy as np
import sys
import os
from PIL import Image
from skimage import color, io
import glob
import shutil

label = sys.argv[1]                  
background = sys.argv[2]        
out = sys.argv[3]

os.makedirs(out, exist_ok=True)
label_path = glob.glob(label+'/*.png')
print(len(label_path))
background_path = glob.glob(background+'/*.png')

# for each_background_path in background_path:
#     name = each_background_path.split('/')[-1]
#     target_path = os.path.join(label, name)
#     each_background = np.array(Image.open(each_background_path).convert("RGB"))
#     each_background = color.rgb2gray(each_background)
#     each_background = (each_background * 255.0).astype('uint8')
#     print(np.unique(each_background))
#     each_background[each_background == 12] = 1     #蓝色
#     each_background[each_background != 1] = 0
#     out_path = os.path.join(out, name)
#     if target_path in label_path:
#         target = Image.open(target_path).convert("RGB")
#         target = np.array(target)
#         target = color.rgb2gray(target)
#         target = (target * 255.0).astype('uint8')
#         print(np.unique(target))
#         target[target == 27] = 2    #红色
#         target[target != 2] = 0
        
#         each_background[target == 2] = 2

#         io.imsave(out_path, each_background)
#     else:
#         print(f'not find {each_background_path}')
#         io.imsave(out_path, each_background)


for i in label_path:
    name = i.split('/')[-1]
    each_background_path = os.path.join(background, name)
    out_path = os.path.join(out, 'Labels', name)
    out_images_path = os.path.join(out, 'Images')
    os.makedirs(out_images_path, exist_ok=True)
    if each_background_path in background_path:
        target = Image.open(i).convert("RGB")
        target = np.array(target)
        target = color.rgb2gray(target)
        target = (target * 255.0).astype('uint8')
        target[target == 27] = 2     #红色
        target[target != 2] = 0
        
        each_background = np.array(Image.open(each_background_path).convert("RGB"))
        each_background = color.rgb2gray(each_background)
        each_background = (each_background * 255.0).astype('uint8')
        each_background[each_background == 12] = 1     #蓝色
        # each_background[each_background == 179] = 0     #灰色
        # each_background[each_background == 255] = 0     #白色
        each_background[each_background != 1] = 0
        
        each_background[target == 2] = 2
        
        io.imsave(out_path, each_background)
        shutil.copy(i.replace('.png', '.tif'), out_images_path)
    else:
        print(f'not find {each_background_path}')
        break