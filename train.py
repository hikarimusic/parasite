import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Yolo_Dataset
from model import Yolov4
from utils import Yolo_Loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_WEIGHT = None
SAVE_WEIGHT = None
EPOCHS = 1
BATCH_SIZE = 64
SUBDIVISION = 32
LEARN_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


def train(Data, Model, Loss):
    Model.to(DEVICE).train()
    Optimizer = torch.optim.SGD(Model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #Optimizer = torch.optim.Adam(Model.parameters(), lr=LEARN_RATE)
    loss_batch = 0
    for batch, data in enumerate(Data):
        image = data[0].to(DEVICE)
        label = data[1].to(DEVICE)
        predict = Model(image)
        loss, loss_box, loss_obj, loss_cls = Loss(predict, label)
        loss.backward()
        loss_batch += loss.item() / (BATCH_SIZE//SUBDIVISION)
        print(f"\rloss: {loss_batch/(batch%SUBDIVISION+1):>7f}  [{(batch+1) * len(label):>5d}/{len(Data.dataset):>5d}]", end="")
        if (batch+1) % SUBDIVISION == 0:
            Optimizer.step()
            Optimizer.zero_grad()
            loss_batch = 0
            print("")


if __name__ == '__main__':
    Dataset = Yolo_Dataset("train")
    Data = DataLoader(Dataset, batch_size=BATCH_SIZE//SUBDIVISION, num_workers=8)
    Model = Yolov4()
    Loss = Yolo_Loss(BATCH_SIZE//SUBDIVISION, DEVICE)
    if LOAD_WEIGHT:
        Model.load_state_dict(torch.load(LOAD_WEIGHT))
    for t in range(EPOCHS):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(Data, Model, Loss)
        if SAVE_WEIGHT:
            torch.save(Model.state_dict(), SAVE_WEIGHT)
