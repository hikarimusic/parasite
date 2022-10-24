import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random

from utils import draw_boxes, show_image


def Padding(image, label, size):
    if image.shape[0] > image.shape[1]:
        scale = size / image.shape[0]
        new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
        displace = (0, int((size - new_size[1]) / 2))
    else:
        scale = size / image.shape[1]
        new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
        displace = (int((size - new_size[0]) / 2), 0)
    image = cv2.resize(image, new_size[::-1])
    image_ = np.zeros((size, size, 3))
    image_[displace[0] : displace[0] + image.shape[0], displace[1] : displace[1] + image.shape[1], :] = image[:, :, :]
    label_ = []
    for box in label:
        box[1] = int(box[1] * scale + displace[1])
        box[2] = int(box[2] * scale + displace[0])
        box[3] = int(box[3] * scale)
        box[4] = int(box[4] * scale)
        label_.append(box)
    return image_, label_


def Augment(image, label):
    # Flip
    flip = random.randint(0, 3)
    if flip == 1:
        image = cv2.flip(image, 0)
    elif flip == 2:
        image = cv2.flip(image, 1)
    elif flip == 3:
        image = cv2.flip(image, -1)
    for i, box in enumerate(label):
        if flip == 1:
            label[i][2] = image.shape[0] - box[2]
        elif flip == 2:
            label[i][1] = image.shape[1] - box[1]
        elif flip == 3:
            label[i][1] = image.shape[1] - box[1]
            label[i][2] = image.shape[0] - box[2]
    # Scale
    scale = int(random.uniform(608, 1216))
    image, label = Padding(image, label, scale)


    return image, label


def Mosaic(images, labels, size):
    image_ = np.zeros((size, size, 3))
    label_ = []
    for i in range(4):
        x = random.randrange(0, size // 4)
        y = random.randrange(0, size // 4)
        x += size // 4 if labels[i][0][2] < images[i].shape[0] // 2 else 0
        y += size // 4 if labels[i][0][1] < images[i].shape[1] // 2 else 0
        cx = images[i].shape[0] // 2 - x
        cy = images[i].shape[1] // 2 - y
        px = size // 2 if i % 2 == 1 else 0
        py = size // 2 if i // 2 == 1 else 0
        ox = cx - px
        oy = cy - py
        s = size // 2
        image_[px:px+s, py:py+s, :] = images[i][cx:cx+s, cy:cy+s, :]
        for b in labels[i]:
            box = [b[0], b[1]-oy, b[2]-ox, b[3], b[4]]
            if box[1] >= py and box[1] < py+s and box[2] >= px and box[2] <= px+s:
                label_ += [box]
        #label_ += [[b[0], b[1]-oy, b[2]-ox, b[3], b[4]] for b in labels[i]] 
    return image_, label_


class Yolo_Dataset(Dataset):
    def __init__(self, path, train):
        super().__init__()
        self.path = path
        self.image = [f for f in os.listdir(self.path) if ".jpg" in f]
        self.image.sort()
        self.label = [f for f in os.listdir(self.path) if ".txt" in f]
        self.label.sort()
        self.train = train

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        if self.train == True:
            indexs = [index]
            indexs += [random.randrange(0, self.__len__()) for i in range(3)]
            images = [cv2.imread(os.path.join(self.path, self.image[i])) / 255. for i in indexs]
            labels = []
            for i, id in enumerate(indexs):
                with open(os.path.join(self.path, self.label[id])) as f:
                    lines = f.readlines()
                label = []
                for box in lines:
                    box = [float(c) for c in box.split()]
                    box[0] = int(box[0])
                    box[1] = int(box[1] * images[i].shape[1])
                    box[2] = int(box[2] * images[i].shape[0])
                    box[3] = int(box[3] * images[i].shape[1])
                    box[4] = int(box[4] * images[i].shape[0])
                    label.append(box)
                labels.append(label)            
            for i in range(4):
                images[i], labels[i] = Augment(images[i], labels[i])
            image, label = Mosaic(images, labels, 608)
            return image, label

        if self.train == False:
            image = cv2.imread(os.path.join(self.path, self.image[index]))
            image = image / 255.
            with open(os.path.join(self.path, self.label[index])) as f:
                lines = f.readlines()
            label = []
            for box in lines:
                box = [float(c) for c in box.split()]
                box[0] = int(box[0])
                box[1] = int(box[1] * image.shape[1])
                box[2] = int(box[2] * image.shape[0])
                box[3] = int(box[3] * image.shape[1])
                box[4] = int(box[4] * image.shape[0])
                label.append(box)
            image, label = Padding(image, label, 608)
            return image, label


if __name__ == '__main__':
    train_dataset = Yolo_Dataset(os.path.join(os.getcwd(), "data", "train"), train=True)
    test_dataset =  Yolo_Dataset(os.path.join(os.getcwd(), "data", "test"), train=False)

    for i in range(10):
        img, lbl = train_dataset.__getitem__(random.randrange(0, 11000))
        img = draw_boxes(img, lbl)
        show_image(img)

    for i in range(10):
        img, lbl = test_dataset.__getitem__(random.randrange(0, 2200))
        img = draw_boxes(img, lbl)
        show_image(img)




    
