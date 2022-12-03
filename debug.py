import os
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import Yolo_Dataset
from model import Yolov4
from utils import Yolo_Loss, Yolo_Detector
from metrics import YOLO_Evaluator
from tools import draw_boxes, show_image


def debug():
    Dataset = Yolo_Dataset("train_raw")
    Data = DataLoader(Dataset, batch_size=1, shuffle=False)
    Model = Yolov4()
    Loss = Yolo_Loss(1, 'cuda')
    Detector = Yolo_Detector(1, 'cuda')
    Evaluator = YOLO_Evaluator(Dataset.classes)

    Model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'result', 'train', 'weights.pt')))
    Model.to('cuda')
    Model.eval()

    img, lbl = Dataset.__getitem__(0)
    img = img.to('cuda')

    for data in Data:

        with torch.no_grad():
            image= data[0].to('cuda')
            label= data[1].to('cuda')
            predict = Model(image)

            '''
            for output in predict:
                batch = output.shape[0]
                size = output.shape[2]
                output = output.reshape(batch, 3, 85, size, size).permute(0, 3, 4, 1, 2)
                for a in range(3):
                    print('size:', size)
                    print('anchor:', a)
                    #print('max:', torch.max(output[0, :, :, i, :]))
                    #print('min:', torch.min(output[0, :, :, i, :]))
                    for i in range(size):
                        for j in range(size):
                            print('  position:', i, j)
                            print('    box:', output[0, i, j, a, :4])
                            print('    obj:', output[0, i, j, a, 4])
                            print('    cls:', output[0, i, j, a, 5:16])                   
            '''
 

                        
            detect = Detector(predict, conf_thresh=0.01)[0]
            #print(detect)
            detect = torch.cat([detect[:, :3], torch.full([detect.shape[0], 2], 1).to('cuda')], dim=1)
            #print(img-image[0])
            #print(torch.max(img))
            #print(torch.max(image[0]))
            img = draw_boxes(img, detect)
            #img = draw_boxes(img, detect)
            show_image(img) 
        return    


if __name__ == '__main__':
    debug()