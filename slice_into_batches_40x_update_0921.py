import os
import re
import skimage.io as io
import glob
import tifffile
import shutil
import openslide
from openslide import deepzoom
import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore")
import time
import datetime
#source_path = "/data/zwt/deeplabv3-plus-pytorch-main/aciduct_aci_stroma/test/"    # 存储待切分大图的文件夹路径
#target_path = "/data/zwt/deeplabv3-plus-pytorch-main/aciduct_aci_stroma/test_patches/"    # 存储切分好的文件夹路径

def run_20X(i,j,real_patchSize,w,h,magnification,filepath,file,source_path,target_pathfold,downsample):
    x = i * real_patchSize
    y = j * real_patchSize
    # 考虑边界情况
    if i  * real_patchSize > w:
        x = w - real_patchSize
    if j * real_patchSize > h:
        y = h - real_patchSize

    if (i  * real_patchSize)+real_patchSize > w:
        x = w - real_patchSize
    if (j * real_patchSize)+real_patchSize > h:
        y = h - real_patchSize

    folder = os.path.exists(target_pathfold)
    if not folder:
        os.makedirs(target_pathfold)
    #print("/data/deeplabv3-plus-pytorch/ndpisplit -Ex{0},z0,{1},{2},{3},{4},{6} -O {7} -cn {5}".format(str(magnification), x, y, real_patchSize, real_patchSize, filepath,str(x)+"_"+str(y)+'_'+str(downsample),target_pathfold))
    os.system("/data/deeplabv3-plus-pytorch/ndpisplit -Ex{0},z0,{1},{2},{3},{4},{6} -O {7} -cn {5}".format(str(magnification), x, y, real_patchSize, real_patchSize, filepath,str(x)+"_"+str(y)+'_'+str(downsample),target_pathfold))
    pics_name = file.split('.')[0]
    output_patchfile=''
    while True:
        #print(target_pathfold+'/'+pics_name+"_x"+str(int(magnification))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif')
        if os.path.exists(target_pathfold+'/'+pics_name+"_x"+str(int(magnification))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif'):
            output_patchfile=target_pathfold+'/'+pics_name+"_x"+str(int(magnification))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif'
            break
        else:
            print('waiting ',target_pathfold+'/'+pics_name+"_x"+str(int(magnification))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif')
            time.sleep(2)
            os.system("/data/deeplabv3-plus-pytorch/ndpisplit -Ex{0},z0,{1},{2},{3},{4},{6} -O {7} -cn {5}".format(str(magnification), x, y, real_patchSize, real_patchSize, filepath,str(x)+"_"+str(y)+'_'+str(downsample),target_pathfold))



def run(i,j,real_patchSize,w,h,magnification,filepath,file,source_path,target_pathfold,downsample):
    x = i * real_patchSize
    y = j * real_patchSize
    # print(x,y,w,h)
    # 考虑边界情况

    if (i  * real_patchSize)+real_patchSize > w:
        x = w - real_patchSize
    if (j * real_patchSize)+real_patchSize > h:
        y = h - real_patchSize                                                  
    
    os.system("/data/deeplabv3-plus-pytorch/ndpisplit -Ex{0},z0,{1},{2},{3},{4},{6} -O {7} -cn {5}".format(str(10), x, y, 512, 512, filepath,str(x)+"_"+str(y)+'_'+str(downsample),target_pathfold))
    pics_name = file.split('.')[0]
    folder = os.path.exists(target_pathfold)
    if not folder:
        os.makedirs(target_pathfold)
    output_patchfile=''
    while True:
        if os.path.exists(target_pathfold+'/'+pics_name+"_x"+str(int(10))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif'):
            output_patchfile=target_pathfold+'/'+pics_name+"_x"+str(int(10))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif'
            break
        else:
            print('waiting ',target_pathfold+'/'+pics_name+"_x"+str(int(10))+'_z0'+'_'+str(x)+"_"+str(y)+'_'+str(downsample)+'.tif')
            time.sleep(2)
            os.system("/data/deeplabv3-plus-pytorch/ndpisplit -Ex{0},z0,{1},{2},{3},{4},{6} -O {7} -cn {5}".format(str(10), x, y, 512, 512, filepath,str(x)+"_"+str(y)+'_'+str(downsample),target_pathfold))
    
def rename(each_tif,real_patchSize,target_pathfold):
    try:
        if int(each_tif.split('.')[0].split('_')[-5].split('x')[-1])==10:
            x=int(each_tif.split('.')[0].split('_')[-3])*4
            y=int(each_tif.split('.')[0].split('_')[-2])*4

        if int(each_tif.split('.')[0].split('_')[-5].split('x')[-1])==20:
            x=int(each_tif.split('.')[0].split('_')[-3])
            y=int(each_tif.split('.')[0].split('_')[-2])
        
        downsample = int(each_tif.split('.')[0].split('_')[-1])
        pics_name = each_tif.split('/')[-1].split('_')[0].split('_')[0]
        new_name = pics_name +" [d=%d,x=%d,y=%d,w=%d,h=%d]"%(int(downsample), int(x), int(y), real_patchSize,real_patchSize)+".tif"
        tile = tifffile.imread(each_tif)
        #print(tile)
        # 4. 下采样
        if int(each_tif.split('.')[0].split('_')[-5].split('x')[-1])==20:
            tile = tile[::downsample, ::downsample]
        
        target_each_pics_path = target_pathfold 
        folder = os.path.exists(target_each_pics_path)
        if not folder:
            os.makedirs(target_each_pics_path)
        image_name_ =os.path.join(target_each_pics_path, pics_name +" [d=%d,x=%d,y=%d,w=%d,h=%d]"%(downsample, x, y, real_patchSize, real_patchSize)+".tif")
        # print(image_name_)
        io.imsave(image_name_, tile)
        os.system("rm -rf {0}".format(each_tif))

    except Exception as e:
        print(e)
        print(each_tif)
        os.system("rm -rf {0}".format(each_tif))
    
def rename_before_slice(folder_path):
    folder_list = os.listdir(folder_path)
    for filename in folder_list:
        cleaned_filename = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', filename)
        cleaned_filename = cleaned_filename.replace('ndpi', '.ndpi')
        new_filepath = os.path.join(folder_path, cleaned_filename)
        old_filepath = os.path.join(folder_path, filename)
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {cleaned_filename}')

def check_slice_into_batches_each_pics(file,target_path): 
    filepath = source_path + "/" + file
    w, h, magnification = 0,0,0
    with tifffile.TiffFile(filepath) as f: 
        for tag in f.pages[0].tags.values():
            name,value = tag.name,tag.value
            if "ImageWidth" in name:
                w = value
            if "ImageLength" in name:
                h = value
            if "65421" in name:
                magnification = value


    if int(magnification)==40:
        downsample = int(magnification/5/2)
    
    if int(magnification)==20:
        downsample = int(magnification/5)
    w =int(w/4)
    h=int(h/4)
    
    # when magnification = 40 downsample = 8; magnification = 20 downsample = 4
    save_patchSize = 512
    real_patchSize = save_patchSize * 1
    
    num_w = math.ceil(np.floor(w/real_patchSize))
    num_h = math.ceil(np.floor(h/real_patchSize))
    count=0
    all_image_list=[]
    for i in range(num_w+1):
        for j in range(num_h+1):

            x = i * real_patchSize
            y = j * real_patchSize
            # print(x,y,w,h)
            # 考虑边界情况
            if (i  * real_patchSize)+real_patchSize > w:
                x = w - real_patchSize
            if (j * real_patchSize)+real_patchSize > h:
                y = h - real_patchSize   

            all_image_list.append(','.join([str(x),str(y),str(w),str(h)]))
    count=len(set(all_image_list))
    each_ndpi_path=target_path+'/'+file.split('.')[0]

    folder = os.path.exists(each_ndpi_path)
    if not folder:
        all_image_list=[]
        os.makedirs(each_ndpi_path)
    else:
        all_image_list = glob.glob(each_ndpi_path+'/*x=*y=*w=*h=*.tif')
    all_image_list=os.listdir(each_ndpi_path) 
    #print(len(all_image_list),count)

    if len(all_image_list)==count:
        return 0,magnification
    else:
        return filepath,magnification


def slice_into_batches_each_pics_20x(file): 
    filepath = source_path + "/" + file
    w, h, magnification = 0,0,0
    print(filepath)
    with tifffile.TiffFile(filepath) as f: 
        for tag in f.pages[0].tags.values():
            name,value = tag.name,tag.value
            if "ImageWidth" in name:
                print(value)
                w = value
            if "ImageLength" in name:
                print(value)
                h = value
            if "65421" in name:
                print(value)
                magnification = value
    
    if int(magnification)==40:
        downsample = int(magnification/5/2)
    
    if int(magnification)==20:
        downsample = int(magnification/5)
    # when magnification = 40 downsample = 8; magnification = 20 downsample = 4
    save_patchSize = 512
    real_patchSize = save_patchSize * downsample
    
    num_w = int(np.floor(w/real_patchSize))
    num_h = int(np.floor(h/real_patchSize))
    pool2 = Pool(10)
    for i in range(num_w +1):
        for j in range(num_h + 1):
            pool2.apply_async(func=run_20X, args=(i,j,real_patchSize,w,h,magnification,filepath,file,source_path,target_path+'/'+file.split('.')[0],downsample))
    pool2.close()
    pool2.join()




def slice_into_batches_each_pics_40x(file): 
    filepath = source_path + "/" + file
    w, h, magnification = 0,0,0
    print(filepath)
    with tifffile.TiffFile(filepath) as f: 
        for tag in f.pages[0].tags.values():
            name,value = tag.name,tag.value

            if "ImageWidth" in name:

                w = value
            if "ImageLength" in name:

                h = value
            if "65421" in name:
                magnification = value

    
    if int(magnification)==40:
        downsample = int(magnification/5/2)
    
    if int(magnification)==20:
        downsample = int(magnification/5)
    w =int(w/4)
    h=int(h/4)
    
    # when magnification = 40 downsample = 8; magnification = 20 downsample = 4
    save_patchSize = 512
    real_patchSize = save_patchSize * 1
    
    num_w = math.ceil(np.floor(w/real_patchSize))
    num_h = math.ceil(np.floor(h/real_patchSize))
    pool2 = Pool(15)
    for i in range(num_w+1):
        for j in range(num_h+1):
            #print(i,j)
            #run(i,j,real_patchSize,w,h,magnification,filepath,file,source_path,target_path+file.split('.')[0],downsample)
            pool2.apply_async(func=run, args=(i,j,real_patchSize,w,h,magnification,filepath,file,source_path,target_path+'/'+file.split('.')[0],downsample))
    pool2.close()
    pool2.join()
    res, _ = check_slice_into_batches_each_pics(file,target_path)
    if res ==0:
       print('success download')
       return 0   
    else:
       print('fail download')
       return 1   


import sys
source_path = sys.argv[1]+'/' #wsis path
target_path = sys.argv[2]+'/' #output path
from  multiprocessing import Process,Pool

rename_before_slice(source_path)

to_do_list=[]
num=0
for file in os.listdir(source_path):
    res,mag= check_slice_into_batches_each_pics(file,target_path)
    # if res!=0 and mag==20:
    if res!=0:
        num=num+1
        #print(num,res)
        to_do_list.append(res)

print(len(to_do_list))
print(to_do_list)


#to_do_list=['1804167C3.ndpi']
#to_do_list = ['/data/syx/xinfuzhu/image///2211808C3.ndpi', '/data/syx/xinfuzhu/image///2210529A1.ndpi', '/data/syx/xinfuzhu/image///2116138C10.ndpi', '/data/syx/xinfuzhu/image///1720501A5.ndpi', '/data/syx/xinfuzhu/image///2106785C12.ndpi', '/data/syx/xinfuzhu/image///19423854.ndpi']
count_num=0
for filepath in to_do_list:
    file = filepath.split('/')[-1]
    print(datetime.datetime.now())
    magnification = 0
    with tifffile.TiffFile(filepath) as f: 
        for tag in f.pages[0].tags.values():
            name,value = tag.name,tag.value
            if "65421" in name:
                magnification = value
    if int(magnification)==40:
        print(magnification)
        print(file)
        slice_into_batches_each_pics_40x(file)
        count_num=count_num+1
    if int(magnification)==20:
        print(magnification)
        print(file)
        slice_into_batches_each_pics_20x(file)
        count_num=count_num+1

real_patchSize=512
pool2 = Pool(10)
count=0
count=0
for each_image_fold in os.listdir(target_path):
    print(each_image_fold,datetime.datetime.now())
    each_image_fold_path =os.path.join(target_path,each_image_fold)
    for each_tif in glob.glob(each_image_fold_path+'/*_*_*_*_*.tif'):
        #rename(each_tif,2048,each_image_fold_path)
        pool2.apply_async(func=rename, args=(each_tif,2048,each_image_fold_path))

pool2.close()
pool2.join()

