#!/usr/bin/python3

import torch
#from unet import UNet
from nets.deeplabv3_plus import DeepLab

def parseToOnnx(pt_path,onxx_path):
    num_classes=2
    backbone="xception"
    downsample_factor=16
    pretrained=False
    net = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    #net = UNet(n_channels=3, n_classes=1)
    #pt_path = '/data/deeplabv3-plus-pytorch/logs/best_epoch_weights.pth'
    net.load_state_dict(
        torch.load(pt_path,
                   map_location=torch.device('cpu')))

    print(net.eval())

    batch_size, channels, height, width = 1, 3, 512, 512
    inputs = torch.randn((batch_size, channels, height, width))

    outputs = net(inputs)
    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'
    #"/data/unet-pytorch-main/logs/best_unet.onnx",  # where to save the model (can be a file or file-like   object)
    torch.onnx.export(
        net,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        onxx_path,
        export_params=
        True,  # store the trained parameter weights inside the model     file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=
        False,  # whether to execute constant folding for optimization
        input_names=['inputs'],  # the model's input names
        output_names=['outputs'],  # the model's output names
        dynamic_axes={
            'inputs': {
                0: 'batch_size'
            },  # variable lenght axes
            'outputs': {
                0: 'batch_size'
            }
        })

    print("ONNX model conversion is complete.")
    return

if __name__ == '__main__':
    import sys
    pt_path= sys.argv[1]
    onxx_path= sys.argv[2]
    print(".onnx")
    parseToOnnx(pt_path,onxx_path)
