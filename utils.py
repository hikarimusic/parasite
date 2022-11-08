import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.grid = []
        self.anchor = []
        self.batch = batch
        self.shape = [76, 38, 19]
        anchors = {
            76 : [[28, 28], [46, 45], [63, 66]] ,
            38 : [[99, 74], [78, 115], [131, 110]] ,
            19 : [[147, 161], [174, 269], [254, 175]]
        }
        for size in self.shape:
            grid_x = torch.linspace(0, size-1, size).view(1, 1, size, 1, 1).repeat(batch, size, 1, 3, 1)
            grid_y = torch.linspace(0, size-1, size).view(1, size, 1, 1, 1).repeat(batch, 1, size, 3, 1)
            self.grid.append(torch.cat([grid_x, grid_y], dim=4)) # [batch, size, size, anchor, xy]
            anchor_w = [torch.full([batch, size, size, 1, 1], anchors[size][i][0]) for i in range(3)]
            anchor_w = torch.cat(anchor_w, dim=3)
            anchor_h = [torch.full([batch, size, size, 1, 1], anchors[size][i][1]) for i in range(3)]
            anchor_h = torch.cat(anchor_h, dim=3)
            self.anchor.append(torch.cat([grid_x, grid_y], dim=4)) # [batch, size, size, anchor, wh]

    def forward(self, output):
        for i in range(3):
            pass

        

if __name__ == '__main__':
    output = [torch.rand(1, 255, 76, 76), torch.rand(1, 255, 38, 38), torch.rand(1, 255, 19, 19)]


    
