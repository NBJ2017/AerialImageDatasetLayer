'''
note:
author: Ruilong Ren
create date: 2018.3.20
updata date: 2018.4.24
'''
import caffe

import numpy as np
from PIL import Image
import os
import random
'''
This layer used to input the Aerial Image Dateset in order to do segment training or test
The Dataset include two files: "trian" - has two files: 'gt' and 'images'
                               "test" - include the test images and each images has no GT labels
updata log: It only could get one sample (image) before this change, and we need batch size(!=1) samples
            in DeepLab task, so I add the param "batch".                    

'''
class AerialImageDatasetLayer(caffe.Layer):
    def setup(self, bottom, top):
        #get the config from the .prototxt file
        params=eval(self.param_str)
        self.dataset_dir = params['dataset_dir']
        self.seed = params.get('seed',None)
        self.split = params['split']
        self.mean = params.get('mean',0)
        self.random = params.get('random',True)
        self.im_w_h = params.get('im_w_h', 505)
        self.batch = params.get('batch',1)

        #check the number of top and bottom
        if len(top) is not 2:
            raise Exception('Need 2 output for this layer!')
        if len(bottom) is not 0:
            raise Exception('Need no input for this layer!')

        #get the image names in dataset_dir/split/images
        self.splitimages_dir = os.path.join(self.dataset_dir, self.split, 'images')
        self.splitlabels_dir = os.path.join(self.dataset_dir, self.split, 'gt')
        self.splitimages_names = list(os.listdir(self.splitimages_dir))
        self.indx = 0
        self.indxs = range(0,self.batch)
        #print(self.indxs)

        #decide whather need to random the images order
        if 'train' not in self.split:
            self.random = False

        if self.random is True:
            random.seed(self.seed)
            for i in range(0,self.batch):
                self.indx = random.randint(0, len(self.splitimages_names)-1)
                self.indxs[i] = self.indx



    def reshape(self, bottom, top):
        #get the image and label
        self.data = self.load_images(self.indxs)
        self.label = self.load_labels(self.indxs)
        #reshape the top shapes
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)


    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.random is True:
            self.indx = random.randint(0, len(self.splitimages_names)-1)
            for i in range(0,self.batch):
                self.indx = random.randint(0, len(self.splitimages_names)-1)
                self.indxs[i] = self.indx
        else:
            # self.indx += self.batch
            #print(self.indx)
            for i in range(0,self.batch):
                self.indx += 1
                self.indxs[i] = self.indx
            if self.indx == len(self.splitimages_names):
                self.indx = 0



    def backward(self, bottom, top, propagate_down):
        pass

    def load_image(self, name):
        image = Image.open(os.path.join(self.splitimages_dir,name))
        #change RGB as BGR
        im = np.array(image,dtype=np.float32)
        im = im[:,:,::-1]
        im -= self.mean
        im = im.transpose([2,0,1]) #change as channel*high*width

        return im

    def load_images(self, indxs):
        imgs = np.zeros([self.batch, 3, self.im_w_h, self.im_w_h],dtype=np.float32)
        for i in range(0,self.batch):
            imgs[i,:,:,:] = self.load_image(self.splitimages_names[indxs[i]])
        return imgs



    def load_label(self, name):
        label = Image.open(os.path.join(self.splitlabels_dir, name))
        label = np.array(label,dtype=np.bool)
        label = label[np.newaxis,:,:,0]  #format is 1*high*width
        label = label.astype(np.uint8)
        #print(np.max(label))

        return label



    def load_labels(self, indxs):
        labels = np.zeros([self.batch, 1, self.im_w_h, self.im_w_h],dtype=np.uint8)
        for i in range(0,self.batch):
            labels[i,:,:,:] = self.load_label(self.splitimages_names[indxs[i]])
        return labels