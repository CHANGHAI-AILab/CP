from multiprocessing import Pool
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import numpy as np 
import openslide
from PIL import Image
import glob

from skimage import io,color
from skimage import io
from skimage.transform import rescale, resize
"""
data = np.ones((9,9))
print(data)
print(data.shape)
x=2
y=4
data.fill(255)
print(data)
data[x:x+2,y:y+2]=data1
print(data)
"""
import sys
Image.MAX_IMAGE_PIXELS=None

import tifffile
def create_each_big_mask(image_name, prediction_masks_path, big_mask_path_name):
    d1,d2=0,0
    w, h, magnification = 0,0,0
    print(image_name,'1111')
    
    wsi = openslide.OpenSlide(image_name)
    properties = wsi.properties
    downsample=0
    for name, value in properties.items():
        if name.startswith('mirax.GENERAL.OBJECTIVE_MAGNIFICATION'):
            magnification =  int(float(value))

        if name.startswith('openslide.level[0].height'):
            d2=int(value)

        if name.startswith('openslide.level[0].width'):
            d1=int(value)

    #print(d1,d2,magnification)
    #downsample = int(80/magnification)
    
    if int(magnification)==40:
        downsample = int(magnification/5/2)
    
    if int(magnification)==20:
        downsample = int(magnification/5)
    print(downsample)
    data = np.ones((int(d2/downsample),int(d1/downsample)))
    print(data.shape)

    all_data=[]
    if len(glob.glob(prediction_masks_path+'/*.tif', recursive=True))==0:
        print('no found prediction',prediction_masks_path)
    else:
        print('save name',big_mask_path_name)
        for path in glob.glob(prediction_masks_path+'/*.tif', recursive=True):
            #try:
            #image = io.imread(path)
            #print(path)

            label = Image.open(path).convert("RGB")
            label = np.array(label)
            label = color.rgb2gray(label)
            image = (label*255.0).astype('uint8')
            image[image==255]=0
            image[image==27]=0
            image[image==91]=255
            for each in np.unique(image):
                if each not in all_data:
                    all_data.append(each)

            #RGBnp = np.uint8(image)
            #RGBnp[RGBnp == 255] = 1
            #image = RGBnp[:,:,1]
            #image   = BinaryNP.fromarray(BinaryNP)
            #image = resize(image, (512,512))
            image_name = path.split('/')[-1]
            image_name = image_name.replace('x=',' ')
            image_name = image_name.replace('y=',' ')
            image_name = image_name.replace('w=',' ')
            image_name = image_name.replace('h=',' ')
            x = int(image_name.split(',')[1].strip())
            y = int(image_name.split(',')[2].strip())
            h = int(image_name.split(',')[3].strip())
            data[int(y/downsample):int(y/downsample)+512,int(x/downsample):int(x/downsample)+512]=image

        data[data==1]=0
        data[data!=0]=1

        #data[data!=0]=1
        data = data.astype(np.uint8)
        #print('save name',big_mask_path_name+'.tiff')
        #tifffile.imwrite(big_mask_path_name+'.tiff',data)

        data[data==1]=255
        data = resize(data, (int(data.shape[0]/10),int(data.shape[1]/10)))
        print(big_mask_path_name+'.jpg')
        io.imsave(big_mask_path_name+'.jpg',data)



origin_path = sys.argv[1]+'/' # where orgin big images saved e.g. /data/zwt/deeplabv3-plus-pytorch-main/changxinghigh/test_20x/
prediction_patch_masks_path = sys.argv[2]+'/' # where patch predictions saved e.g. /data/zwt/deeplabv3-plus-pytorch-main/changxinghigh_predictions/
bigimages = os.listdir(origin_path)
countnum_=0
for bigimg in bigimages:    # each folder of each big image
    print(countnum_,bigimg)
    try:
        
        bigimg_name =bigimg.split('.')[0]
        if '.mrxs' not in bigimg:
            continue
        image_name_path = origin_path + bigimg
        big_mask_path_name = prediction_patch_masks_path + bigimg_name + "_predictions"
        prediction_masks_path = prediction_patch_masks_path + bigimg_name + '/'
        create_each_big_mask(image_name_path, prediction_masks_path, big_mask_path_name)
        countnum_=countnum_+1
    except Exception as e:
        print(e)