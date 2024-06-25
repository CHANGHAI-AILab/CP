# import os 
# import random
# import shutil
# import sys

# input_root = sys.argv[1]
# out_root = sys.argv[2]
# downsmaple_number = int(sys.argv[3])

# input_label_path = os.path.join(input_root, 'Labels')
# out_label_path = os.path.join(out_root, 'Labels')
# input_image_path = os.path.join(input_root, 'Images')
# out_image_path = os.path.join(out_root, 'Images')

# input_label_name = os.listdir(input_label_path)


# name = [i.replace('.png','') for i in input_label_name]


# sample_name_list = random.sample(name, downsmaple_number)

# output_image_list = [os.path.join(input_image_path, i+'.tif') for i in sample_name_list]
# output_label_list = [os.path.join(input_label_path, i+'.png') for i in sample_name_list]

    
# if not os.path.exists(out_label_path):
#     os.makedirs(out_label_path)

# if not os.path.exists(out_image_path):
#     os.makedirs(out_image_path)

# for label in output_label_list:
#     shutil.copy(label, out_label_path)
#     print(f'copy {label}')

# for image in output_image_list:
#     shutil.copy(image, out_image_path)
#     print(f'copy {image}')


import os 
import random
import shutil
import sys
from multiprocessing import Pool

def copy_files(args):
    input_image_path, out_image_path, input_label_path, out_label_path = args
    shutil.copy(input_image_path, out_image_path)
    shutil.copy(input_label_path, out_label_path)

def main():
    input_root = sys.argv[1]
    out_root = sys.argv[2]
    downsmaple_number = int(sys.argv[3])

    input_label_path = os.path.join(input_root, 'Labels')
    out_label_path = os.path.join(out_root, 'Labels')
    input_image_path = os.path.join(input_root, 'Images')
    out_image_path = os.path.join(out_root, 'Images')

    input_label_name = os.listdir(input_label_path)
    name = [i.replace('.png', '') for i in input_label_name]
    sample_name_list = random.sample(name, downsmaple_number)

    output_image_list = [os.path.join(input_image_path, i + '.tif') for i in sample_name_list]
    output_label_list = [os.path.join(input_label_path, i + '.png') for i in sample_name_list]

    if not os.path.exists(out_label_path):
        os.makedirs(out_label_path)

    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)

    args_list = []
    for i in range(len(output_image_list)):
        args_list.append((output_image_list[i], os.path.join(out_image_path, sample_name_list[i] + '.tif'),
                          output_label_list[i], os.path.join(out_label_path, sample_name_list[i] + '.png')))

    with Pool() as p:
        p.map(copy_files, args_list)

    print('Copy completed.')

if __name__ == '__main__':
    main()
    
