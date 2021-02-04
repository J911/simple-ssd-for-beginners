





import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os
import cv2
from config import opt

from lib.augmentations import preproc_for_test, preproc_for_train

VOC_LABELS = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    )

CUSTOMDB_LABELS = ('어린이 보호', '직진금지', '좌회전 및 U턴', '좌회전', '통행금지', '우회로', '우회전', '회전형 교차로', '철길건널목', '직진')


class VOCDetection(data.Dataset):


    def __init__(self, opt, image_sets=[['2007', 'trainval'], ['2012', 'trainval']], is_train=True):


        #你的voc root
        self.root = opt.VOC_ROOT
        #使用的数据集列表,每个数据集包括年份和使用的部分
        self.image_sets = image_sets
        self.is_train = is_train
        self.opt = opt    #我们需要知道在预处理的时候要将图片resize到多大以及减去的方差等信息
    
        self.ids = []
        #遍历数据集将图片得路径加到id里面
        for (year, name) in self.image_sets:
            root_path = os.path.join(self.root, 'VOC' + year)
            ano_file = os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')
    
            with open(ano_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    ano_path = os.path.join(root_path, 'Annotations', line + '.xml')
                    img_path = os.path.join(root_path, 'JPEGImages', line + '.jpg')
                    self.ids.append((img_path, ano_path))



    
    
    def __getitem__(self, index):
        img_path, ano_path = self.ids[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        boxes, labels = self.get_annotations(ano_path)
        
        if self.is_train:
            image, boxes, labels = preproc_for_train(image, boxes, labels, opt.min_size, opt.mean)
            image = torch.from_numpy(image)
           
        
        
        target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)
        
        return image, target



    def get_annotations(self, path):
        
        tree = ET.parse(path)

        #得到真实坐标和标签
        boxes = []
        labels = []
        
        for child in tree.getroot():
            if child.tag != 'object':
                continue

            bndbox = child.find('bndbox')
            box =[
                float(bndbox.find(t).text) - 1
                for t in ['xmin', 'ymin', 'xmax', 'ymax']
            ]


            label = VOC_LABELS.index(child.find('name').text) 
            
            boxes.append(box)
            labels.append(label)


        return np.array(boxes), np.array(labels)
            

        

    def __len__(self):
        return len(self.ids)
        
class CustomDetection(data.Dataset):
    def __init__(self, opt, root, dbtype='train'):
        self.root = root
        self.type = dbtype
        self.label_dir = self.root + '/' + self.type + '/'
        self.image_dir = self.root + '/images/'

        self.labels = sorted(glob.glob(self.label_dir + '*'))
        
        self.ids = []
        
        for label in self.labels:
            self.ids.append((self.image_dir + label[len(self.label_dir):-4] + '.png', label))
        print(self.ids)
    def __getitem__(self, index):
        anno_path = self.labels[index]
        img_path = self.image_dir + anno_path[len(self.label_dir):-4] + '.png'

        boxes, labels = self.get_annotations(anno_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.type == 'train':
            image, boxes, labels = preproc_for_train(image, boxes, labels, opt.min_size, opt.mean)
            image = torch.from_numpy(image)
            
        if len(labels) == 0:
            target = np.array([])
        else:
            target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)
        return image, target

        
    def get_annotations(self, path):
        cls_ids = []
        bboxs = []

        labels = open(path)
        line = labels.readline()

        while line:
            label = [float(x) for x in line.split(' ')]

            xmin = (label[1] * 800) - (label[3] * 800 / 2)
            ymin = (label[2] * 240) - (label[4] * 240 / 2)

            xmax = (label[1] * 800) + (label[3] * 800 / 2)
            ymax = (label[2] * 240) + (label[4] * 340 / 2)

            cls_id = int(label[0])
            bbox = [xmin, ymin, xmax, ymax]

            cls_ids.append(cls_id)
            bboxs.append(bbox)

            line = labels.readline()
        return np.array(bboxs, dtype=np.float64), np.array(cls_ids)

    def __len__(self):
        return len(self.labels)