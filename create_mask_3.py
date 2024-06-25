# from multiprocessing import Pool, Lock
# import cv2 as cv
# import numpy as np
# import sys
# import os

# def create_mask(image_path, label_path):
#     image = os.listdir(image_path)
#     label = os.listdir(label_path)
#     # print(label_path)
#     for each_image_name in image:
#         each_image_path = os.path.join(image_path, each_image_name)
#         each_image = cv.imread(each_image_path, 0)
#         # print(each_image.shape)
#         # print(min(np.unique(each_image)))
#         # print(max(np.unique(each_image)))
#         t, dst = cv.threshold(each_image, 200, 255, cv.THRESH_BINARY_INV)

#         kernel = np.ones((20, 20), np.uint8)
#         dst = cv.dilate(dst, kernel, 4)
        
#         kernel = np.ones((40, 40), np.uint8)
#         dst = cv.erode(dst, kernel, 4)
        
#         contours, hierarchy = cv.findContours(dst,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#         cv.drawContours(dst,contours,-1,(255),-1)
#         for each_contour in contours:
#             area = cv.contourArea(each_contour)
#             if area < 200:
#                 cv.drawContours(dst, [each_contour], -1, 0, -1)
        
#         # cv.namedWindow(f"{each_image_name}", cv.WINDOW_NORMAL)
#         # cv.imshow(f"{each_image_name}", dst)
#         # cv.waitKey(0)
        
#         dst[dst == 255] = 1
#         dst[dst != 1] = 0
#         # print(np.unique(dst))
#         label_name = each_image_name.split('.')[0] + '.png'
#         if label_name in label:
#             each_label_path = os.path.join(label_path, label_name)
#             each_label = cv.imread(each_label_path, 0)
#             # print(np.unique(each_label))
#             dst[each_label==1] = 2
#             # for i in range(each_label.shape[0]):
#             #     for j in range(each_label.shape[1]):
#             #         if each_label[i, j] == 1:
#             #             dst[i, j] = 2
#             # print('+'*10)
#             # print(np.unique(dst))
#             print(f'create {each_image_name} mask')
#             cv.imwrite(each_label_path, dst)
        
        

# # image_path = '/mnt/e/data/yxa0831/test/image/'
# # label_path = '/mnt/e/data/yxa0831/test/label/'
# image_path = sys.argv[1]
# label_path = sys.argv[2]

# # create_mask(image_path, label_path)

# pool = Pool(10)
# # lock = Lock()
# # print(image_path)
# pool.apply_async(func=create_mask, args=(image_path, label_path))
# pool.close()
# pool.join()
                    
    
from multiprocessing import Pool, Lock
import cv2 as cv
import numpy as np
import sys
import os

def create_mask_positive_annotaion(image_path, label_path):
    image = [i for i in os.listdir(image_path) if i.endswith('.tif')]
    label = os.listdir(label_path)
    label_name_1 = label_path.split('/')[-1]
    output = label_path.replace(label_name_1, 'new_mask')
    os.makedirs(output, exist_ok=True)
    for each_image_name in image:
        each_image_path = os.path.join(image_path, each_image_name)
        each_image = cv.imread(each_image_path, 0)
        t, dst = cv.threshold(each_image, 175, 255, cv.THRESH_BINARY_INV)
        # print(np.unique(dst))
        kernel = np.ones((1, 1), np.uint8)
        dst = cv.dilate(dst, kernel, 4)
        
        kernel = np.ones((2, 2), np.uint8)
        dst = cv.erode(dst, kernel, 4)
        
        contours, hierarchy = cv.findContours(dst,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(dst,contours,-1,(255),-1)
        for each_contour in contours:
            area = cv.contourArea(each_contour)
            if area < 100:
                cv.drawContours(dst, [each_contour], -1, 0, -1)
                
        print(np.unique(dst))
        
        dst[dst == 255] = 1
        dst[dst != 1] = 0
        label_name = each_image_name.split('.')[0] + '.png'
        if label_name in label:
            each_label_path = os.path.join(label_path, label_name)
            each_label = cv.imread(each_label_path, 0)
            print(np.unique(each_label))
            dst[each_label!=0] = 2
            print(f'create {each_image_name} mask')
        each_label_path = os.path.join(output, label_name)
        print(each_label_path)
        print(np.unique(dst))
        #dst[dst==1]=100
        #dst[dst==2]=255
        cv.imwrite(each_label_path, dst)
        #break



def create_mask_negative(image_path):
    image = [i for i in os.listdir(image_path) if i.endswith('.tif')]
    label = os.listdir(image_path)
    label_name_1 = image_path.split('/')[-1]
    output = image_path.replace(label_name_1, 'new_mask')
    os.makedirs(output, exist_ok=True)
    for each_image_name in image:
        each_image_path = os.path.join(image_path, each_image_name)
        each_image = cv.imread(each_image_path, 0)
        t, dst = cv.threshold(each_image, 175, 255, cv.THRESH_BINARY_INV)
        # print(np.unique(dst))
        kernel = np.ones((1, 1), np.uint8)
        dst = cv.dilate(dst, kernel, 4)
        
        kernel = np.ones((2, 2), np.uint8)
        dst = cv.erode(dst, kernel, 4)
        
        contours, hierarchy = cv.findContours(dst,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(dst,contours,-1,(255),-1)
        for each_contour in contours:
            area = cv.contourArea(each_contour)
            if area < 100:
                cv.drawContours(dst, [each_contour], -1, 0, -1)
                
        print(np.unique(dst))
        
        dst[dst == 255] = 1
        dst[dst != 1] = 0
        label_name = each_image_name.split('.')[0] + '.png'
        if label_name in label:
            each_label_path = os.path.join(image_path, label_name)
            each_label = cv.imread(each_label_path, 0)
            print(np.unique(each_label))
            dst[each_label!=0] = 2
            print(f'create {each_image_name} mask')
        each_label_path = os.path.join(output, label_name)
        print(each_label_path)
        print(np.unique(dst))
        #dst[dst==1]=100
        #dst[dst==2]=255
        cv.imwrite(each_label_path, dst)
        #break
        
        
image_path = sys.argv[1]
label_path = sys.argv[2]

pool = Pool(10)
pool.apply_async(func=create_mask_positive_annotaion, args=(image_path, label_path))
#create_mask_negative(image_path)
#pool.apply_async(func=create_mask_negative, args=(image_path))

pool.close()
pool.join()
                    
    
    
