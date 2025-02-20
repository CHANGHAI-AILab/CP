For training

python slice_into_batches_40x_update_0921.py /data/xb/project_name /data/xb/project_name_patch
CUDA_VISIBLE_DEVICES=2 nohup python train_sys_para.py 3 xception True model_data/deeplab_xception.pth project_name logproject_name /data/deeplabv3-plus-pytorch/tmpproject_name 1,1,7 >> project_name.log 2>&1 &

For testing and visualization

python predict_batches_update_0916.py /data/deeplabv3-plus-pytorch/logproject_name /data/deeplabv3-plus-pytorch/project_name/test_patch/  /data/deeplabv3-plus-pytorch/logproject_name/test_patch_prediction/  0:9000
python create_big_mask_batches_3_classes_40x_2048_from114_mrsx.py /data/deeplabv3-plus-pytorch/logproject_name/test/  /data/deeplabv3-plus-pytorch/logproject_name/test_patch_prediction/

All CP model checkpoint share in Baiduyun https://pan.baidu.com/s/1hT-uS7KL3msQn4GXLxhgeg?pwd=qcpx 
