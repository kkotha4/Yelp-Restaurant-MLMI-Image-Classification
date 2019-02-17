import sys
import random
import os,numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from scipy.misc import imread, imresize
from skimage.transform import resize
from scipy.sparse import csr_matrix
from PIL import Image
import pandas as pd
import cv2
import ast
import numpy as np
from collections import defaultdict
VOC_CLASSES = ("good_for_lunch","good_for_dinner","takes_reservations","outdoor_seating","restaurant_is_expensive","has_alcohol",
                "has_table_service","ambience_is_classy","good_for_kid")

class preprocess(data.Dataset):

    def __init__(self, data_path, transform,csv_file,random_crops=0):
        self.data_path = data_path
        self.transform = transform
        self.random_crops = random_crops

        self.csv_file=csv_file
        self.images, self.labels, self.business_id = self.__retrieve_data()
        print(self.images)
        print(self.labels)
        print(self.business_id)
        #print(self.labels)
        #print(self.label_order)


    def __getitem__(self,index):
        
        x = cv2.imread(os.path.join(self.data_path,self.images[index]),1)
        x= cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        x = Image.fromarray(x)

        scale = np.random.rand() * 2 + 0.25
        w = int(x.size[0] * scale)
        h = int(x.size[1] * scale)
        if min(w, h) < 227:
            scale = 227 / min(w, h)
            w = int(x.size[0] * scale)
            h = int(x.size[1] * scale)

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.labels[index]
        z = self.business_id[index]
        
        #print(self.images[index])
        return x, y, z

    def __len__(self):
        return len(self.images)

    def __retrieve_data(self):
        #train_dir='./train_photos_new'
        train_imgs = os.listdir(self.data_path)
        image_list=[]
        label_list=[]
        business_id=[]
        for index,i in enumerate(train_imgs):
            if index%1000 == 0:
                print("%0.4f percent is completed" %(index*100/len(train_imgs)))
            #print(i)
            new=ast.literal_eval(self.csv_file[self.csv_file["photo_id"]==i]["labels_new"].item())
            id_=self.csv_file[self.csv_file["photo_id"]==i]["business_id"].item()

            #print(new)
            label_list.append(new)
            #train_dir = './train_photos_new'
            #img = cv2.imread(os.path.join(self.data_path,i),1)
            image_list.append(i)
            business_id.append(id_)
        return image_list,np.array(label_list).astype(np.float32),business_id
    
class batch_sampling(data.Sampler):
     
    def __init__(self,batch_size, drop_last, business_id):
        '''if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))'''
        #self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.business_id=business_id
        self.business_index=defaultdict(list)
        for v, k in enumerate(self.business_id):
              self.business_index[k].append(v)

            
        
       
        self.unique_business_id=list(set(self.business_id))
        
    
    def __iter__(self):
        self.business_id
        batch = []
        business=random.choice(self.unique_business_id)            
        for idx in self.business_index[business]:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
            

    def __len__(self):
        if self.drop_last:
            return len(self.business_id) // self.batch_size
        else:
            return (len(self.business_id) + self.batch_size - 1) // self.batch_size

        
class DummySampler(data.Sampler):
    def __init__(self, business_id):
        self.business_id=business_id
        self.business_index=defaultdict(list)
        for v, k in enumerate(self.business_id):
              self.business_index[k].append(v)
        self.unique_business_id=list(set(self.business_id))

    def __iter__(self):
        business=random.choice(self.unique_business_id)  
        print(business)
        print ('\tcalling Sampler:__iter__')
        #random_list
        return iter(self.business_index[business])

    def __len__(self):
        print ('\tcalling Sampler:__len__')
        return len(self.business_index[business])
