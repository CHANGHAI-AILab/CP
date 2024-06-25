import os
import re
import shutil
import subprocess
from subprocess import Popen, PIPE, STDOUT

import numpy as np

NUMBER_OF_MAGNIFICATIONS = 5
MAX_TILE_SIZE = 20000  # 20000*20000*3 = 1.2 GB RAM
COMPRESSION_NONE = 'n'
COMPRESSION_JPEG = 'j'

"""

class NDPI:

    def __init__(self, filepath, debug=False):
        if not filepath.endswith('.ndpi'):
            raise IOError(os.path.basname(filepath) + ' is not an NDPI file.')
        elif not os.path.exists(filepath):
            raise IOError(filepath + ' does not exist.')
        else:
            self.filepath = filepath
            self.readInfoFromHeader()
        self.commandLine = []
        self.debug = debug


    def __repr__(self):
        s = ['Nanozoomer Digital Pathology Image']
        s.append(' Source lens: x%d' % self.sourceLens)
        s.append(' Size: %d x %d pixels' % (self.size[0], self.size[1]))
        s.append(' Resolution: %d x %d pixels/cm2' % (self.resolution[0], self.resolution[1]))
        s.append(' Pixel spacing: %.3f x %.3f nm' % (self.spacing[0]*1e6, self.spacing[1]*1e6))
        return '\n'.join(s)


    def readInfoFromHeader(self):
        cmd = ['tifftopnm', '-headerdump', self.filepath]
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
        text = process.communicate()[0]

        pattern = r'Image Width: (\d+) Image Length: (\d+)'
        numbersStrings = re.findall(pattern, text)
        self.size = np.array([int(n) for tup in numbersStrings for n in tup])

        pattern = r'Tag 65421: (\d+)'
        numbersStrings = re.findall(pattern, text)
        self.sourceLens = int(numbersStrings[0])
        self.magnifications = [self.sourceLens/4.**i for i in range(NUMBER_OF_MAGNIFICATIONS)]
        self.magnifications = [int(x) if x%1 == 0 else x for x in self.magnifications]  # E. g. 20.0 -> 20

        pattern = r'Resolution: (\d+), (\d+) pixels/cm'
        numbersStrings = re.findall(pattern, text)
        resolution = np.array([int(n) for tup in numbersStrings for n in tup])
        self.resolution = resolution
        self.spacing = 10./resolution  # 10 mm


    def getMagnificationsString(self):
        s = ['x%s'%n for n in self.magnifications]
        return ', '.join(s)


    def split(self, magnifications=[], outputPath=None, compression=COMPRESSION_JPEG, run=True):
        commandLine = ['ndpisplit']
        if magnifications:
            if not isinstance(magnifications, list):
                magnificationsString = '-x' + str(magnifications)
            else:
                magnificationsString = '-x'
                for magnification in magnifications:
                    magnificationsString += str(magnification) + ','

            commandLine.append(magnificationsString.strip(','))
        commandLine.append('-c' + compression)
        commandLine.append(self.filepath)
        self.commandLine = commandLine
        if run:
            self.run()
            tiffPath = self.getMagnificationFilepath(magnifications)
            if outputPath is None:  # we suppose there is only one magnification
                outputPath = tiffPath
            else:
                src, dst = tiffPath, outputPath
                if os.path.exists(src):
                    shutil.move(src, dst)
            return outputPath


    def extractROI(self, magnification, topLeftX, topLeftY, width, height, outputPath=None, compression=COMPRESSION_JPEG, run=True):
        if magnification not in self.magnifications:
            raise ValueError('Magnification x%s is not available. Choose between the following: %s' % (magnification, self.getMagnificationsString()))
        magnificationSize = self.getSize(magnification)
        inX = topLeftX < magnificationSize[0] and width <= magnificationSize[0] and (topLeftX + width) <= magnificationSize[0] + 1
        inY = topLeftY < magnificationSize[1] and height <= magnificationSize[1] and (topLeftY + height) <= magnificationSize[1] + 1

        commandLine = ['ndpisplit']
        options = '-Ex%s,%d,%d,%d,%d' % (str(magnification), topLeftX, topLeftY, width, height)
        commandLine.append(options)
        commandLine.append('-c' + compression)
        commandLine.append(self.filepath)
        # print magnificationSize
        # print topLeftX, topLeftY, width, height
        # print commandLine
        # print inX
        # print inY
        if not (inX and inY):
            raise ValueError('Wrong ROI size. The image size for this magnification is %s.' % self.getSizeString(magnificationSize))
        self.commandLine = commandLine
        if run:
            self.run()
            roiPath = self.getROIFilepath(magnification)
            if outputPath is None:
                return roiPath
            else:
                src, dst = roiPath, outputPath
                if os.path.exists(src):
                    shutil.move(src, dst)
                    return dst


    def getSizeString(self, size):
        return '%d x %d' % (size[0], size[1])


    def getAffine(self, magnification=None, sz=0.05, oz=0):
        if magnification is None:
            magnification = self.sourceLens
        sx, sy = self.getSpacing(magnification)
        affine = np.diag([sx, sy, sz, 1])
        affine[2, 3] = oz
        sizeX, sizeY = self.getSize(magnification)
        affine[0, 3] = sizeX * sx
        affine[0, 0] *= -1
        return affine


    def extractROIFromFile(self, acsvPath, magnification=None, outputDir=None, prefix=None, oz=0, flipX=False, flipY=False, removeTIFF=True, makeNifti=True):
        roiCenterWorld, roiSizeWorld = acsv.ROI(acsvPath).getCenterAndSize()
        self.extractROIFromCenterAndSize(roiCenterWorld, roiSizeWorld, magnification=magnification, outputDir=outputDir, prefix=prefix, oz=oz, flipX=flipX, flipY=flipY, removeTIFF=removeTIFF, makeNifti=makeNifti)


    def extractROIFromCenterAndSize(self, roiCenterWorld, roiSizeWorld, magnification=None, outputDir=None, prefix=None, oz=0, flipX=False, flipY=False, removeTIFF=True, makeNifti=True):
        import nibabel as nib
        import acsv
        import utils

        if magnification is None:
            magnification = self.sourceLens
        ijk2ras = self.getAffine(magnification)

        ras2ijk = np.linalg.inv(ijk2ras)
        roiCenterPixel = nib.affines.apply_affine(ras2ijk, roiCenterWorld)
        sx, sy, sz, _ = np.diag(ijk2ras)
        roiSizePixel = np.abs(roiSizeWorld / np.array((sx, sy, sz)))

        topLeft = np.round(roiCenterPixel - roiSizePixel/2).astype(int)
        topRight = np.round(roiCenterPixel + roiSizePixel/2).astype(int)
        topRightX = topRight[0]
        topLeftX, topLeftY, _ = topLeft
        width, height, _ = np.round(roiSizePixel).astype(int)

        if flipX:
            topLeftX = self.flip(topRightX, 0, magnification)
            topRightX = self.flip(topLeftX, 0, magnification)
        if flipY:
            topLeftY = self.flip(topLeftY, 1, magnification) - height


        numTilesX = width / MAX_TILE_SIZE + 1
        numTilesY = height / MAX_TILE_SIZE + 1
        numTiles = numTilesX * numTilesY

        tileWidth = width / numTilesX
        tileHeight = height / numTilesY

        if outputDir is None:
            outputDir = os.path.dirname(self.filepath)
        if prefix is None:
            prefix = os.path.splitext(os.path.basename(self.filepath))[0]

        for tileY in range(numTilesY):
            for tileX in range(numTilesX):
                tileTopLeftX = topLeftX + tileX*tileWidth
                tileTopLeftY = topLeftY + tileY*tileHeight
                tileTopRightX = tileTopLeftX + tileWidth - 1

                if flipX:
                    tileColumn = numTilesX - tileX - 1
                else:
                    tileColumn = tileX

                if flipY:
                    tileRow = numTilesY - tileY - 1
                else:
                    tileRow = tileY

                if numTiles == 1:
                    outputPath = os.path.join(outputDir, prefix + '.tif')
                else:
                    outputPath = os.path.join(outputDir, prefix + '_tile_%d_%d.tif' % (tileRow, tileColumn))

                utils.ensureDir(outputPath)
                roiPath = self.extractROI(magnification, tileTopLeftX, tileTopLeftY, tileWidth, tileHeight, outputPath=outputPath)

                if makeNifti:
                    import ImageUtils as iu
                    roiAffine = ijk2ras[:]
                    roiAffine[:3, :3] = np.abs(roiAffine[:3, :3])
                    roiAffine[0, 3] = self.flip(tileTopRightX, 0, magnification) * roiAffine[0, 0]
                    roiAffine[1, 3] = tileTopLeftY * roiAffine[1, 1]

                    if flipX:
                        roiAffine[0, 0] *= -1
                        roiAffine[0, 3] = abs(roiAffine[0, 0]) * (tileTopLeftX + tileWidth)
                    if flipY:
                        roiAffine[1, 1] *= -1
                        roiAffine[1, 3] = abs(roiAffine[1, 1]) * self.flip(tileTopLeftY, 1, magnification)

                    roiAffine[2, 3] = oz
                    iu.histologyImageToNiftiRGB(roiPath, affine=roiAffine)

                if removeTIFF:
                    os.remove(roiPath)


    def flip(self, n, dim, magnification=None):
        if magnification is None:
            magnification = self.getSourceLens()
        sizeDim = self.getSize(magnification)[dim]
        return sizeDim - n - 1


    def getMagnificationFilepath(self, magnification):
        base = os.path.splitext(self.filepath)[0]
        magPath = '{}_x{}_z0.tif'.format(base, magnification)
        return magPath


    def getROIFilepath(self, magnification):
        base = os.path.splitext(self.filepath)[0]
        magPath = '{}_x{}_z0_1.tif'.format(base, magnification)
        return magPath


    def getSize(self, magnification=None):
        if magnification is None:
            magnification = self.sourceLens
        return map(int, self.size / (self.sourceLens / magnification))


    def getSpacing(self, magnification=None):
        if magnification is None:
            magnification = self.sourceLens
        return self.spacing * (self.sourceLens / magnification)


    def getSourceLens(self):
        return self.sourceLens


    def getROISize(self, acsvPath):
        import acsv
        _, _, width, height = acsv.ROI(acsvPath).getCropValuesWorld()
        spacingX, spacingY = self.getSpacing()
        widthPixels = np.round(width / spacingX).astype(int)
        heightPixels = np.round(height / spacingY).astype(int)
        return widthPixels, heightPixels


    def getROIMemory(self, acsvPath):
        widthPixels, heightPixels = self.getROISize(acsvPath)
        numPixels = widthPixels * heightPixels
        numBytes = numPixels * 3
        if numBytes > 1e9:
            print '%.3f' % (numBytes/1e9), 'GB'
        else:
            print '%d' % (numBytes/1e6), 'MB'


    def getSplitRatio(self, remove=True):
        import ImageUtils as iu
        lowestMagnification = self.magnifications[-2]  # Sometimes the lowest magnification image is truncated
        imgPath = self.split(lowestMagnification, compression=COMPRESSION_NONE)
        splitRatio = iu.getSplitLeftRightColumnRatio(imgPath)
        if remove:
            os.remove(imgPath)
        return splitRatio


    def printCommandLine(self):
        print ' '.join(self.commandLine)


    def run(self):
        if self.debug:
            print 'Running:'
            self.printCommandLine()
        subprocess.call(self.commandLine)




def getSplitRatio(filepath, remove=True):
    ndpiFile = NDPI(filepath)
    return ndpiFile.getSplitRatio(remove=remove)

"""
"""
cmd = ['tifftopnm', '-headerdump', "/data/zwt/deeplabv3-plus-pytorch-main/net/test/1815873.ndpi"]
process = Popen(cmd, stdout=PIPE, stderr=STDOUT)
text = process.communicate()[0]

pattern = r'Image Width: (\d+) Image Length: (\d+)'
numbersStrings = re.findall(pattern, text)
self.size = np.array([int(n) for tup in numbersStrings for n in tup])

pattern = r'Tag 65421: (\d+)'
numbersStrings = re.findall(pattern, text)
self.sourceLens = int(numbersStrings[0])
self.magnifications = [self.sourceLens/4.**i for i in range(NUMBER_OF_MAGNIFICATIONS)]
self.magnifications = [int(x) if x%1 == 0 else x for x in self.magnifications]  # E. g. 20.0 -> 20

pattern = r'Resolution: (\d+), (\d+) pixels/cm'
numbersStrings = re.findall(pattern, text)
resolution = np.array([int(n) for tup in numbersStrings for n in tup])
self.resolution = resolution
self.spacing = 10./resolution  # 10 mm
"""
"""
import tifffile


with tifffile.TiffFile("/data/zwt/deeplabv3-plus-pytorch-main/net/test/1815873.ndpi") as f: 
    for tag in f.pages[0].tags.values():
        name,value = tag.name,tag.value
        if "ImageWidth" in name:
            print(value)
        if "ImageLength" in name:
            print(value)
"""

from skimage import io,color
import multiprocessing
import numpy as np
import glob
#os.system("rm -rf *z=2*")
#os.system("rm -rf *z=1*")

from PIL import Image
import random
def run(each_label,label_path_save,PREPROCESSTYPE):
    #if "1408261A1 [d=4,x=7680,y=7680,w=2048,h=2048]" in each_label:
    print(each_label)
    label = Image.open(each_label).convert("RGB")
    label = np.array(label)
    #label = io.imread(each_label)
    label = color.rgb2gray(label)
    label = (label*255.0).astype('uint8')
    tar1 = np.unique(label)

    # three class 
    if PREPROCESSTYPE==3:
        label[label==12]=1
        label[label==180]=0
        label[label==255]=0
        label[label>1]=2
        tar2 = np.unique(label)


    # two class 
    if PREPROCESSTYPE==2:
        #PREPROCESS TYPE2
        label[label==12]=0
        label[label==180]=0
        label[label==255]=0
        label[label>1]=1
        tar2 = np.unique(label)


    print(tar1,tar2)
    label = Image.fromarray(np.uint8(label))
    if PREPROCESSTYPE==3:
        if 2 in tar2:
            os.makedirs(label_path_save+'/'+'label',exist_ok=True)
            label.save(label_path_save+'/'+'label/'+each_label.split('/')[-1])
        else:
            os.makedirs(label_path_save+'/'+'background',exist_ok=True)
            label.save(label_path_save+'/'+'background/'+each_label.split('/')[-1])


    if PREPROCESSTYPE==2:
        if 1 in tar2:
            os.makedirs(label_path_save+'/'+'label',exist_ok=True)
            label.save(label_path_save+'/'+'label/'+each_label.split('/')[-1])
        else:
            os.makedirs(label_path_save+'/'+'background',exist_ok=True)
            label.save(label_path_save+'/'+'background/'+each_label.split('/')[-1])

    #label.save(label_path_save+'/'+each_label.split('/')[-1])
    return tar1,tar2
def run3(total_label_path,each_main_label):
    #print(each_main_label)
    each_label_name_z1 = os.path.join(total_label_path,each_main_label.split('/')[-1].replace('z=2','z=1'))
    each_orignal_label = os.path.join(total_label_path,each_main_label.split('/')[-1].replace(',z=2',''))
    each_label_name_z2  =  os.path.join(total_label_path,each_main_label.split('/')[-1])
    
    print('each_label_name_z2',each_label_name_z2)
    print('each_label_name_z1',each_label_name_z1)
    print('each_orignal_label',each_orignal_label)


    labelz1=None
    labelz2=None
    label=None

    print(each_orignal_label)

    if os.path.exists(each_label_name_z1):
        labelz1 = io.imread(each_label_name_z1)
        print("get z1 data")

    if os.path.exists(each_label_name_z2):
        print("get z2 data")
        labelz2 = io.imread(each_label_name_z2)
    
    if os.path.exists(each_orignal_label):
        print("get oring data")
        label = io.imread(each_orignal_label)

    #print(labelz1)

    if os.path.exists(each_orignal_label):
        print("find orignl label")
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if os.path.exists(each_label_name_z1):
                    if labelz1[i][j] == 2:
                        label[i][j]=2
                elif os.path.exists(each_label_name_z2): 
                    if labelz2[i][j] ==2 :
                        label[i][j]=2
                else:
                    pass
        label0=label
        label0 = Image.fromarray(np.uint8(label0))
        label0.save(each_orignal_label)
        print("SAVE PATH ", each_orignal_label)
        label = io.imread(each_orignal_label)
    
    if os.path.exists(each_label_name_z1):
        print("find z1 label")
        for i in range(labelz1.shape[0]):
            for j in range(labelz1.shape[1]):
                if os.path.exists(each_orignal_label):
                    if label[i][j] == 2:
                        labelz1[i][j]=2
                elif os.path.exists(each_label_name_z2):
                    if labelz2[i][j] ==2 :
                        labelz1[i][j]=2
                else:
                    pass
        label0=labelz1
        label0 = Image.fromarray(np.uint8(label0))
        label0.save(each_orignal_label)
        print("SAVE PATH ", each_orignal_label)
        label = io.imread(each_orignal_label)

    if os.path.exists(each_label_name_z2):
        print("find z2 label")
        for i in range(labelz2.shape[0]):
            for j in range(labelz2.shape[1]):
                if os.path.exists(each_orignal_label):
                    if label[i][j] == 2:
                        labelz2[i][j]=2
                elif os.path.exists(each_label_name_z1):
                    if labelz1[i][j] ==2 :
                        labelz2[i][j]=2
                else:
                    pass
        label0=labelz2
        label0 = Image.fromarray(np.uint8(label0))
        print("SAVE PATH ", each_orignal_label)
        label0.save(each_orignal_label)
    


    if os.path.exists(each_label_name_z2):
        os.remove(each_label_name_z2)
        print("remove z2 label ",each_label_name_z2)
    
    
    if os.path.exists(each_label_name_z1):
        os.remove(each_label_name_z1)
        print("remove z1 label ",each_label_name_z1)
    
    
    
    each_image_name_z1 = os.path.join(total_image_path,each_main_label.split('/')[-1].replace('z=2','z=1').replace('png','tif'))
    each_image_name_z2 = each_image_name_z1.replace('z=1','z=2')
    orginal_image_name = each_image_name_z2.replace(',z=2','')
    
    
    if os.path.exists(each_image_name_z2) and 'z=2' in each_image_name_z2:
        print('rename',each_image_name_z2,orginal_image_name)
        os.rename(each_image_name_z2,orginal_image_name)
        print("remove z2 image",each_image_name_z2)

    if os.path.exists(each_image_name_z1) and 'z=1' in each_image_name_z1:
        os.rename(each_image_name_z1,orginal_image_name)
        print('rename',each_image_name_z1,orginal_image_name)
        print("remove z1 image",each_image_name_z1)


def run2(total_label_path,each_main_label):
    #print(each_main_label)
    each_label_name_z2 = os.path.join(total_label_path,each_main_label.split('/')[-1].replace('z=1','z=2'))
    each_orignal_label = os.path.join(total_label_path,each_main_label.split('/')[-1].replace(',z=1',''))
    each_label_name_z1  =  os.path.join(total_label_path,each_main_label.split('/')[-1])
    
    print('each_label_name_z2',each_label_name_z2)
    print('each_label_name_z1',each_label_name_z1)
    print('each_orignal_label',each_orignal_label)


    labelz1=None
    labelz2=None
    label=None

    print(each_orignal_label)

    if os.path.exists(each_label_name_z1):
        labelz1 = io.imread(each_label_name_z1)
        print("get z1 data")

    if os.path.exists(each_label_name_z2):
        print("get z2 data")
        labelz2 = io.imread(each_label_name_z2)
    
    if os.path.exists(each_orignal_label):
        print("get oring data")
        label = io.imread(each_orignal_label)

    #print(labelz1)

    if os.path.exists(each_orignal_label):
        print("find orignl label")
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if os.path.exists(each_label_name_z1):
                    if labelz1[i][j] == 2:
                        label[i][j]=2
                elif os.path.exists(each_label_name_z2): 
                    if labelz2[i][j] ==2 :
                        label[i][j]=2
                else:
                    pass
        label0=label
        label0 = Image.fromarray(np.uint8(label0))
        label0.save(each_orignal_label)
        print("SAVE PATH ", each_orignal_label)
        label = io.imread(each_orignal_label)
    
    if os.path.exists(each_label_name_z1):
        print("find z1 label")
        for i in range(labelz1.shape[0]):
            for j in range(labelz1.shape[1]):
                if os.path.exists(each_orignal_label):
                    if label[i][j] == 2:
                        labelz1[i][j]=2
                elif os.path.exists(each_label_name_z2):
                    if labelz2[i][j] ==2 :
                        labelz1[i][j]=2
                else:
                    pass
        label0=labelz1
        label0 = Image.fromarray(np.uint8(label0))
        label0.save(each_orignal_label)
        print("SAVE PATH ", each_orignal_label)
        label = io.imread(each_orignal_label)

    if os.path.exists(each_label_name_z2):
        print("find z2 label")
        for i in range(labelz2.shape[0]):
            for j in range(labelz2.shape[1]):
                if os.path.exists(each_orignal_label):
                    if label[i][j] == 2:
                        labelz2[i][j]=2
                elif os.path.exists(each_label_name_z1):
                    if labelz1[i][j] ==2 :
                        labelz2[i][j]=2
                else:
                    pass
        label0=labelz2
        label0 = Image.fromarray(np.uint8(label0))
        print("SAVE PATH ", each_orignal_label)
        label0.save(each_orignal_label)
    


    if os.path.exists(each_label_name_z2):
        os.remove(each_label_name_z2)
        print("remove z2 label ",each_label_name_z2)
    
    
    if os.path.exists(each_label_name_z1):
        os.remove(each_label_name_z1)
        print("remove z1 label ",each_label_name_z1)
    
    
    
    each_image_name_z2 = os.path.join(total_image_path,each_main_label.split('/')[-1].replace('z=1','z=2').replace('png','tif'))
    each_image_name_z1 = each_image_name_z2.replace('z=2','z=1')
    orginal_image_name = each_image_name_z2.replace(',z=2','')
    
    
    if os.path.exists(each_image_name_z2) and 'z=2' in each_image_name_z2:
        print('rename',each_image_name_z2,orginal_image_name)
        os.rename(each_image_name_z2,orginal_image_name)
        print("remove z2 image",each_image_name_z2)

    if os.path.exists(each_image_name_z1) and 'z=1' in each_image_name_z1:
        os.rename(each_image_name_z1,orginal_image_name)
        print('rename',each_image_name_z1,orginal_image_name)
        print("remove z1 image",each_image_name_z1)

def rename_before_ndpi(path):
    all_png = glob.glob(path+'/*/*/*.png')
    all_tif = glob.glob(path+'/*/*/*.tif')
    all = all_png + all_tif
    special_chars = str.maketrans('&()', '---')
    for file_path in all:
        new_name = file_path.translate(special_chars)
        os.rename(file_path, new_name)
        print(f'rename {new_name}')
            
if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    number_class = int(sys.argv[2])
    rename_before_ndpi(dataset_path)

    #dataset_path = "/data/deeplabv3-plus-pytorch/xlathree"
    total_image_path=dataset_path+'/Images'
    total_label_path=dataset_path+'/Labels'
    """
    #选图挑背景
    imagepath ="/media/ubuntu/timchen2/yxa/tumor_down4/Images"
    count =1
    #选择奇数列的背景采样
    for each in os.listdir("/media/ubuntu/timchen2/yxa/tumor_down4/Labels_three_class/background")[::2]:
        imagename =each.replace('.png','.tif')
        imagename =imagename.replace(' ','\\ ')
        print(imagename)
        print("cp -r {0} {1}".format(imagepath+'/'+imagename,"/media/ubuntu/MyPassport/background_filter"))
        os.system("cp -r {0} {1}".format(imagepath+'/'+imagename,"/media/ubuntu/MyPassport/background_filter"))
    """

    #dataset_path ='/media/ubuntu/Seagate2/changxinghigh'

    for each_image in os.listdir(dataset_path):
        each_image =each_image.replace(' ','\\ ')
        each_image_path = os.path.join(dataset_path,each_image)
        #print(each_image_path)
        os.makedirs(total_image_path,exist_ok=True)
        os.makedirs(total_label_path,exist_ok=True)
        image= each_image_path+'/Images'
        label= each_image_path+'/Labels'
        #print("cp -r {0} {1}".format(image+'/*',total_image_path))
        os.system("cp -r {0} {1}".format(image+'/*',total_image_path))
        os.system("cp -r {0} {1}".format(label+'/*',total_label_path))

    label_path_save= total_label_path+'cov'
    os.makedirs(label_path_save,exist_ok=True)

    with multiprocessing.Pool(processes=3) as pool:
        got1 = []
        got2 = []
        PREPROCESSTYPE=number_class
        for each_label in glob.glob(total_label_path+'/*'):
            #run(each_label,label_path_save)
            result = pool.apply_async(run, (each_label,label_path_save,PREPROCESSTYPE)) # 非阻塞方法
            #print("*"*18)
            print(result)
            tar1 = result.get()[0]
            tar2 = result.get()[1]
            for each1 in tar1:
                got1.append(each1)

            for each2 in tar2:
                got2.append(each2)

        pool.close()
        pool.join()
        print("Sub-process done.")
        print('we have this type in dataset: ',set(got1),'convert to ',set(got2))
    os.rename(total_label_path,total_label_path+'_old')
    os.rename(label_path_save,total_label_path+'_preprocess')

    #选择如何采样的模式：：2奇数列

    background_images =total_label_path+'_preprocess/background'
    label_images = total_label_path+'_preprocess/label'
    os.makedirs(total_label_path,exist_ok=True)
    backgroud_list_images = os.listdir(background_images)
    #print(len(random.sample(backgroud_list_images,len(os.listdir(label_images)))))
    """
    count=1
    for imagename in backgroud_list_images:
        #for imagename in random.sample(backgroud_list_images,len(os.listdir(label_images))):
        imagename =imagename.replace(' ','\\ ')
        count=count+1
        #print(count)
        #print(imagename)
        #print("cp -r {0} {1}".format(background_images+'/'+imagename,total_label_path))
        os.system("cp -r {0} {1}".format(background_images+'/'+imagename,total_label_path))
    #print("cp -r {0} {1}".format(label_images+'/*',total_label_path))
    os.system("cp -r {0} {1}".format(label_images+'/*',total_label_path))
    """

    count=1
    if len(os.listdir(label_images)) > len(backgroud_list_images):
        for imagename in backgroud_list_images:
            imagename =imagename.replace(' ','\\ ')
            count=count+1
            os.system("cp -r {0} {1}".format(background_images+'/'+imagename,total_label_path))
    else:
        for imagename in random.sample(backgroud_list_images,len(os.listdir(label_images))):
            imagename =imagename.replace(' ','\\ ')
            count=count+1
            os.system("cp -r {0} {1}".format(background_images+'/'+imagename,total_label_path))

    for each in os.listdir(label_images):
        each =each.replace(' ','\\ ')
        os.system("cp -r {0} {1}".format(label_images+'/'+each,total_label_path))

    os.rename(total_image_path,total_image_path+'_old')
    os.makedirs(total_image_path,exist_ok=True)
    for each in os.listdir(total_label_path):
        imagename =each.replace('.png','.tif')
        imagename =imagename.replace(' ','\\ ')
        #print(imagename)
        #print("cp -r {0} {1}".format(total_image_path+'_old'+'/'+imagename,total_image_path))
        os.system("cp -r {0} {1}".format(total_image_path+'_old'+'/'+imagename,total_image_path))

    from skimage import io
    with multiprocessing.Pool(processes=15) as pool:
        for each_main_label in glob.glob(total_label_path+"/*z=1*"):
            #print(each_main_label)
            #print(each_main_label)
            #run2(total_label_path,each_main_label)
            pool.apply_async(run2, (total_label_path,each_main_label)) # 非阻塞方法
        pool.close()
        pool.join()

    from skimage import io
    with multiprocessing.Pool(processes=15) as pool:
        for each_main_label in glob.glob(total_label_path+"/*z=2*"):
            #print(each_main_label)
            #print(each_main_label)
            #run2(total_label_path,each_main_label)
            pool.apply_async(run3, (total_label_path,each_main_label)) # 非阻塞方法
        pool.close()
        pool.join()

    #from skimage import io
    #for each_main_label in glob.glob(total_label_path+"/*z=1*"):
        #print(each_main_label)

        """
        each_label_name_z2 = os.path.join(total_label_path,each_main_label.split('/')[-1].replace('z=1','z=2'))
        each_orignal_name = os.path.join(total_label_path,each_main_label.split('/')[-1].replace(',z=1',''))
        labelz1 = io.imread(each_main_label)

        if os.path.exists(each_label_name_z2):
            labelz2 = io.imread(each_label_name_z2)

        if os.path.exists(each_orignal_name):
            label0 = io.imread(each_orignal_name)

        for i in range(labelz1.shape[0]):
            for j in range(labelz1.shape[1]):
                if labelz1[i][j] == 2:
                    label0[i][j]=2
                elif os.path.exists(each_label_name_z2) and labelz2[i][j] ==2 :
                    label0[i][j]=2
                else:
                    pass
        label0 = Image.fromarray(np.uint8(label0))
        print('save',each_orignal_name)
        label0.save(each_orignal_name)

        if os.path.exists(each_label_name_z2):
            os.remove(each_label_name_z2)
        os.remove(each_main_label)

        each_image_name_z2 = os.path.join(total_image_path,each_main_label.split('/')[-1].replace('z=1','z=2').replace('png','tif'))
        each_image_name_z1 = each_image_name_z2.replace('z=2','z=1')
        
        if os.path.exists(each_image_name_z2):
            os.remove(each_image_name_z2)
        os.remove(each_image_name_z1)
        """
