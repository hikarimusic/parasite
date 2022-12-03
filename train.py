import os
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd

from dataset import Yolo_Dataset
from model import Yolov4
from utils import Yolo_Loss, Yolo_Detector
from metrics import YOLO_Evaluator


def train(Opt):
    Dataset = Yolo_Dataset("train")
    Data = DataLoader(Dataset, batch_size=Opt.batch_size//Opt.subdivision, num_workers=Opt.workers, shuffle=True, drop_last=True)
    Model = Yolov4()
    Loss = Yolo_Loss(Opt.batch_size//Opt.subdivision, Opt.device)
    Detector = Yolo_Detector(Opt.batch_size//Opt.subdivision, Opt.device)
    Evaluator = YOLO_Evaluator(Dataset.classes)
    
    if os.path.exists(os.path.join(Opt.dir, 'weights.pt')):
        Model.load_state_dict(torch.load(os.path.join(Opt.dir, 'weights.pt')))
    if Opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(Model.parameters(), lr=Opt.learn_rate, momentum=Opt.momentum, weight_decay=Opt.weight_decay)
    elif Opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(Model.parameters(), lr=Opt.learn_rate)
    if os.path.exists(os.path.join(Opt.dir, 'results.pt')):
        results = pd.read_csv(os.path.join(Opt.dir, 'results.pt')).to_dict(orient='list')
    else:
        results = {"loss": [], "loss_box": [], "loss_obj": [], "loss_cls": [], "recall": [], "precision": [], "mAP50": [], "mAP50:95": []}

    status = {"loss": 0, "loss_box": 0, "loss_obj": 0, "loss_cls": 0, "TP": 0, "FP": 0, "FN": 0}
    for epoch in range(Opt.epochs):
        print(f'\nEpoch {epoch+1}\n========')
        print('training\n--------')
        print(f'{"progress":>13} {"step":>11} {"current":>9} {"loss":>9} {"loss_box":>9} {"loss_obj":>9} {"loss_cls":>9} {"TP":>9} {"FP":>9} {"FN":>9}')
        Model.to(Opt.device).train()
        for batch, data in enumerate(Data):
            image = data[0].to(Opt.device)
            label = data[1].to(Opt.device)
            predict = Model(image)
            #with torch.no_grad():
            #    detection = Detector(predict)
            #    Evaluator.process(detection, label)            
            loss, loss_box, loss_obj, loss_cls = Loss(predict, label)
            loss.backward()
            if (batch+1) % Opt.subdivision == 0:
                optimizer.step()
                optimizer.zero_grad()                
            status["loss"] = (status["loss"] * batch + loss.item()) / (batch + 1)
            status["loss_box"] = (status["loss_box"] * batch + loss_box.item()) / (batch + 1)
            status["loss_obj"] = (status["loss_obj"] * batch + loss_obj.item()) / (batch + 1)
            status["loss_cls"] = (status["loss_cls"] * batch + loss_cls.item()) / (batch + 1)
            #status["TP"] = Evaluator.counts["TP"]
            #status["FP"] = Evaluator.counts["FP"]
            #status["FN"] = Evaluator.counts["FN"]            
            logger = [f'\r[{(batch+1)*len(label):>5}/{len(Data.dataset):>5}]']
            logger += [f'[{(batch+1)//Opt.subdivision:>4}/{len(Data)//Opt.subdivision:>4}]']
            logger += [f'{loss.item():>9}'[:9]]
            logger += [f'{status["loss"]:>9}'[:9], f'{status["loss_box"]:>9}'[:9], f'{status["loss_obj"]:>9}'[:9], f'{status["loss_cls"]:>9}'[:9]]
            logger += [f'{status["TP"]:>9}'[:9], f'{status["FP"]:>9}'[:9], f'{status["FN"]:>9}'[:9]]
            logger = ' '.join(logger)
            print(logger, end="")
        torch.save(Model.state_dict(), os.path.join(Opt.dir, 'weights.pt'))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=os.path.join(os.getcwd(), 'result', 'train'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--subdivision', type=int, default=32)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD')
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    Opt = parse_opt()
    train(Opt)