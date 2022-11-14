import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.grid = []
        self.anchor = []
        self.batch = batch
        self.size = [76, 38, 19]
        anchors = {
            76 : [[28, 28], [46, 45], [64, 66]] ,
            38 : [[102, 74], [78, 115], [132, 113]] ,
            19 : [[149, 163], [174, 268], [257, 176]]
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        for size in self.size:
            grid_x = torch.linspace(0, size-1, size).view(1, 1, size, 1, 1).repeat(batch, size, 1, 3, 1)
            grid_y = torch.linspace(0, size-1, size).view(1, size, 1, 1, 1).repeat(batch, 1, size, 3, 1)
            self.grid.append(torch.cat([grid_x, grid_y], dim=4).to(self.device)) # [batch, size, size, anchor, xy]
            anchor_w = [torch.full([batch, size, size, 1, 1], anchors[size][i][0]) for i in range(3)]
            anchor_w = torch.cat(anchor_w, dim=3)
            anchor_h = [torch.full([batch, size, size, 1, 1], anchors[size][i][1]) for i in range(3)]
            anchor_h = torch.cat(anchor_h, dim=3)
            self.anchor.append(torch.cat([anchor_w, anchor_h], dim=4).to(self.device)) # [batch, size, size, anchor, wh]

    
    def NMS(self, boxes, scores, conf_thresh, nms_thresh):
        xy_1 = boxes[:, :2] - boxes[:, 2:] / 2
        xy_2 = boxes[:, :2] + boxes[:, 2:] / 2
        area = boxes[:, 2] * boxes[:, 3]
        confs, order = torch.sort(scores, descending=True)
        order = order[confs > conf_thresh]
        keep = []
        while order.shape[0] > 0:
            keep.append(order[0])
            idx_a = order[0:1].repeat(order.shape[0])
            idx_b = order
            inter_1 = torch.maximum(xy_1[idx_a, :], xy_1[idx_b, :])
            inter_2 = torch.minimum(xy_2[idx_a, :], xy_2[idx_b, :])
            inter = torch.clamp(inter_2 - inter_1, min=0)
            outer_1 = torch.minimum(xy_1[idx_a, :], xy_1[idx_b, :])
            outer_2 = torch.maximum(xy_2[idx_a, :], xy_2[idx_b, :])
            outer = torch.clamp(outer_2 - outer_1, min=0)
            area_i = inter[:, 0] * inter[:, 1]
            area_u = area[idx_a] + area[idx_b] - area_i
            r2 = torch.pow(boxes[idx_a, :1] - boxes[idx_b, :1], 2).sum(dim=1)
            c2 = torch.pow(outer, 2).sum(dim=1)
            diou = area_i / area_u - r2 / c2
            order = order[diou < nms_thresh]
        return keep


    def forward(self, x, nms=False, conf_thresh=0.4, nms_thresh=0.6):
        predict = []
        for i in range(3):
            output = x[i].reshape(self.batch, 3, 85, self.size[i], self.size[i]).permute(0, 3, 4, 1, 2) # [batch, size, size, anchor, channel]
            output[:, :, :, :, 0:2] = torch.sigmoid(output[:, :, :, :, 0:2]) * 1.05 - 0.025 + self.grid[i] # scale_x_y = 1.05
            output[:, :, :, :, 0:2] *= 608 // self.size[i]
            output[:, :, :, :, 2:4] = torch.exp(output[:, :, :, :, 2:4]) * self.anchor[i]
            output[:, :, :, :, 4: ] = torch.sigmoid(output[:, :, :, :, 4: ])
            output = output.reshape(self.batch, self.size[i]*self.size[i]*3, 85) # [batch, n_box, channel]
            predict.append(output)
        predict = torch.cat(predict, dim=1)
        if nms == True:
            predict[:, :, 5:] *= predict[:, :, 4:5].repeat(1, 1, 80)
            predict_ = []
            for b in range(self.batch):
                boxes = []
                for c in range(80):
                    keep = self.NMS(predict[b, :, :4], predict[b, :, c+5], conf_thresh, nms_thresh)
                    boxes.append(torch.cat([torch.full([len(keep), 1], c).to(self.device), predict[b, keep, :4]], dim=1))
                predict_.append(torch.cat(boxes, dim=0))                
            predict = predict_
        return predict


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(batch=16)
    input_ = [((torch.rand(16, 255, s, s)-0.5)*1.09).to(device) for s in [76, 38, 19]]
    output_ = detector(input_, nms=True)

    import random
    from dataset import Yolo_Dataset, draw_boxes, show_image
    test_dataset =  Yolo_Dataset("train_raw")
    for i in range(16):
        img, lbl = test_dataset.__getitem__(random.randrange(0, 11000))
        img = draw_boxes(img, output_[i])
        show_image(img)    

