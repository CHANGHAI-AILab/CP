import numpy as np
import sys
import os
from PIL import Image
from  multiprocessing import Pool
from skimage import color, io
import shutil

def process_image(image_path, out_path, image):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    img_array = color.rgb2gray(img_array)
    img_array = (img_array * 255.0).astype('uint8')
    img_array[img_array == 12] = 1     #蓝色
    img_array[img_array == 179] = 0     #灰色
    # img_array[img_array == 42] = 2     #红色
    img_array[img_array == 255] = 0     #白色
    img_array[img_array != 1] = 0    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    print(f'complete {image}')
    io.imsave(os.path.join(out_path, image), img_array)
    
def copy(label_path, label_tmp, background):
    find_list = os.listdir(label_path)
    label_tmp_list = os.listdir(label_tmp)
    
    for each_label in find_list:
        each_label_path = os.path.join(label_path, each_label)
        label = Image.open(each_label_path)
        label = np.array(label)
        if 1 in label and each_label in label_tmp_list:
            each_tmp_label_path = os.path.join(label_tmp, each_label)
            each_tmp_label = np.array(Image.open(each_tmp_label_path))
            each_tmp_label[label == 1] = 2
            io.imsave(each_label_path, each_tmp_label)
            print(f'save {each_label}')
            
    for  each_tmp in label_tmp_list:
        if not os.path.isdir(background):
            os.mkdir(background)
        if each_tmp not in find_list:
            shutil.copy(os.path.join(label_tmp, each_tmp), background)
            print(f'copy {each_tmp}')
            
            
label_path = sys.argv[1]        #原图
label_tmp = sys.argv[2]         #原图转化成2类的临时文件
label = sys.argv[3]             #最上层的目标类
background = sys.argv[4]        #背景类

label_list = os.listdir(label_path)
pool1 = Pool(10)
for each_label in label_list:
    each_label_path = os.path.join(label_path, each_label)
    # process_image(each_label_path, label_tmp, each_label)
    pool1.apply_async(func=process_image , args=(each_label_path, label_tmp, each_label))
pool1.close()
pool1.join()

pool2 = Pool(10)
pool2.apply_async(func=copy, args=(label, label_tmp, background))
# copy(label, label_tmp, background)
pool2.close()
pool2.join()

 
