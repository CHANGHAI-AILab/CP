#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image
import os
import re
from tqdm import tqdm
from  multiprocessing import Process,Pool

from deeplab import DeeplabV3

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    trained_model_path = sys.argv[1]   
    # where best_epoch_weights.pth saved e.g. /data/zwt/deeplabv3-plus-pytorch-main/logs_changxinglow/
    origin_path = sys.argv[2]
    save_path   = sys.argv[3]
    rangeid   = sys.argv[4]
    # dir_origin_path = "/data/zwt/datasets/Slice/after_slicing/"
    # dir_save_path   = "/data/zwt/datasets/Slice/prediction/"

    #trained_model = trained_model_path + "best_epoch_weights.pth"
    #shutil.copy(trained_model, "/data/deeplabv3-plus-pytorch/model_data/")
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    
    def run_f(origin_path,bigimg,save_path,save_list,modelpath):
        print("we reeealy got check_point"+modelpath+'/best_epoch_weights.pth')
        deeplab = DeeplabV3(modelpath+'/best_epoch_weights.pth')
        dir_origin_path = origin_path + bigimg + '/'
        dir_save_path = save_path + bigimg + '/'
        #folder = os.path.exists(dir_save_path)
        #if not folder:
        #    os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        # print(len(img_names))
        count=0
        for img_name in img_names:
            if not check_predicted(img_name, save_list,dir_origin_path,dir_save_path):
                run_p(img_name,dir_origin_path,dir_save_path,deeplab)
            else:
                print('yes')


    def run_p(img_name,dir_origin_path,dir_save_path,deeplab):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            image       = Image.open(image_path)
            r_image     = deeplab.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name))
            
            
    def check_predicted(img_name, save_list,dir_origin_path,dir_save_path):
        # if img_name in save_list and len(os.listdir(dir_origin_path))==len(os.listdir(dir_save_path)):
        if img_name in save_list:
            if len(img_name) == len(os.path.join(dir_save_path, img_name)):
                return True
            else:
                return False
        else:
            return False
        


    if mode == "predict":
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(deeplab.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    # elif mode == "dir_predict":
    #     import pandas as pd 
    #     alllist=[]
    #     data = pd.read_excel("/data/deeplabv3-plus-pytorch/jianzhi/肿瘤切片名称17.18.xls")
    #     count=0
    #     for each in list(data['切片号']):
    #         cleaned_filename =  re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', each)
    #         alllist.append(cleaned_filename)
            
    
        
    #     pool2 = Pool(8)

    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     save_list = os.listdir(save_path)

    #     count=0

    #     i, j = int(rangeid.split(':')[0]), int(rangeid.split(':')[1])
    #     print("+++++++", i, "::::", j, "+++++++++++")
    #     bigimages = os.listdir(origin_path)[i:j]

    #     print(len(bigimages))
    #     print(len(alllist))
    #     bigimages = list(set(bigimages) & set(alllist))
    #     print(len(bigimages))

    #     for bigimg in bigimages:    # each folder of each big image
    #         #print(bigimg)
    #         if  bigimg in save_list and len(os.listdir(origin_path+'/'+bigimg))==len(os.listdir(save_path+'/'+bigimg)):
    #             pass
    #             #print('done')
    #         else:
    #             print('to do')
    #             print(bigimg)
    #             print(count)
    #             count=count+1
    #             #run_f(origin_path,bigimg,save_path,save_list,trained_model_path)
    #             pool2.apply_async(func=run_f, args=(origin_path,bigimg,save_path,save_list,trained_model_path))
    #     pool2.close()
    #     pool2.join()
    #     #print(count)

    elif mode == "dir_predict":
        
        pool2 = Pool(4)
        i,j=int(rangeid.split(':')[0]),int(rangeid.split(':')[1])
        print("+++++++",i,"::::",j,"+++++++++++")
        bigimages = os.listdir(origin_path)[i:j]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_list = os.listdir(save_path)
        print(bigimages)
        for bigimg in bigimages:    # each folder of each big image
            print(bigimg)
            #run_f(origin_path,bigimg,save_path,save_list,trained_model_path)
            pool2.apply_async(func=run_f, args=(origin_path,bigimg,save_path,save_list,trained_model_path))
        pool2.close()
        pool2.join()
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
