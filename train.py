from dataset import Yolo_Dataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    dataset = Yolo_Dataset("train")
    dataload = DataLoader(dataset, num_workers=16, batch_size=16)
    size = dataset.__len__()

    for batch, (X, y) in enumerate(dataload):
        #print(f"\r[{(batch+1) * len(X):>5d}/{size:>5d}]", end="")
        print(f"[{(batch+1) * len(X):>5d}/{size:>5d}]")

    
