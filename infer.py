#!/usr/bin/env python
#coding=utf-8
import numpy as np
from PIL import Image

import caffe
import cv2
import os
from timer import Timer

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#image_path_name='/home/irsa/tf_unet/data/AerialImageDataset/test/images/bellingham1.tif'
image_path_name='/media/irsa/RRL/研二/fcn_building_segment/Level20/test.tif'
image_name=os.path.splitext(os.path.basename(image_path_name))[0]
#image_path=os.path.split(image_path_name)[0]
im = Image.open(image_path_name)
in_ = np.array(im, dtype=np.float32)
in_ = in_[500:1000,500:1000,:]
im_original_clip = in_.astype(np.uint8)
cv2.imwrite(image_name+'.tif',in_.astype(np.uint8))
print(in_)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))


# load net
net = caffe.Net('/home/irsa/deeplab/exper/building/config/deeplab_largeFOV/deploy.prototxt', '/home/irsa/deeplab/exper/building/model/deeplab_largeFOV/train_iter_20000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
timer = Timer()
timer.tic()
net.forward()
timer.toc()
print ('Segment took {:.3f}s ').format(timer.total_time)
#out = net.blobs['score'].data[0].argmax(axis=0)
out = net.blobs['score'].data[0].argmax(axis=0)

#sava output result (class_n*im_x*im_y)
print(net.blobs['score'].data[0].shape)
print(out)
np.savetxt('result.txt',out)
out=np.asarray(out,dtype=np.float32)
out = out[0:-5,0:-5]
print(type(out))
print(len(out),len(out[1]))


#set dict in order to inditate the different colors
COLOR={4:(255 ,193, 193),
       2: (174,238,238),
       3: (255,218,185),
       1: (0,0,128),
       5: (102, 205 ,170),
       6: (144 ,238 ,144),
       7: (132 ,112, 255),
       8: (0 ,139 ,69),
       9:(175 ,238 ,238),
       10: (192 ,255, 62),
        11: (0 ,255 ,255),
        12: (238 ,230, 133),
        13: (0 ,255, 0),
        14: (238, 238, 0),
        15: (205 ,92, 92),
       16: (245 ,245, 245),
17: (	255, 20 ,147),
18: (238 ,44 ,44),
19: (148 ,0 ,211),
20: (144 ,238 ,144)}
#give a new array to save segment image
image_seg=np.zeros([len(out),len(out[1]),3])
image_seg = im_original_clip
#base on different classes give the block different colors
for class_indx in np.arange(1,21):
    this_class_indx=np.where(out==class_indx)
    img_seg = image_seg[this_class_indx[0],this_class_indx[1],:].astype(np.float32)*0.2
    #image_seg[this_class_indx[0],this_class_indx[1],:] += np.around(np.array(COLOR[class_indx],dtype=np.float32)*0.6).astype(image_seg.dtype)
    img_seg += np.array(COLOR[class_indx],dtype=np.float32)*0.8
    image_seg[this_class_indx[0],this_class_indx[1],:] = np.round(img_seg).astype(image_seg.dtype)


# image_seg[:,:,0]=out/20*225
# image_seg[:,:,1]=255/2
# image_seg[:,:,2]=255/2

# cv2.imwrite(os.path.join(image_path,image_name+'_seg.jpg'),image_seg)
cv2.imwrite(image_name+'_Deeplabseg.tif',image_seg)
