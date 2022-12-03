import torch
from torch import nn


class Yolo_Detector(nn.Module):
    def __init__(self, batch, device):
        super().__init__()
        self.batch = batch
        self.device = device
        self.grid = {}
        self.anchor = {}
        anchor_dict = {
            76 : [[28, 28], [46, 45], [64, 66]] ,
            38 : [[102, 74], [78, 115], [132, 113]] ,
            19 : [[149, 163], [174, 268], [257, 176]]
        }
        for size in anchor_dict.keys():
            grid_x = torch.linspace(0, size-1, size).view(1, 1, size, 1, 1).repeat(batch, size, 1, 3, 1)
            grid_y = torch.linspace(0, size-1, size).view(1, size, 1, 1, 1).repeat(batch, 1, size, 3, 1)
            grid = torch.cat([grid_x, grid_y], dim=4)
            anchor_w = [torch.full([batch, size, size, 1, 1], anchor_dict[size][i][0]) for i in range(3)]
            anchor_w = torch.cat(anchor_w, dim=3)
            anchor_h = [torch.full([batch, size, size, 1, 1], anchor_dict[size][i][1]) for i in range(3)]
            anchor_h = torch.cat(anchor_h, dim=3)
            anchor = torch.cat([anchor_w, anchor_h], dim=4)
            self.grid[size] = (grid.to(self.device))
            self.anchor[size] = (anchor.to(self.device))

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
            inter = torch.clamp(inter_2 - inter_1, 0)
            outer_1 = torch.minimum(xy_1[idx_a, :], xy_1[idx_b, :])
            outer_2 = torch.maximum(xy_2[idx_a, :], xy_2[idx_b, :])
            outer = torch.clamp(outer_2 - outer_1, min=0)
            area_i = inter[:, 0] * inter[:, 1]
            area_u = area[idx_a] + area[idx_b] - area_i
            r2 = torch.pow(boxes[idx_a, :2] - boxes[idx_b, :2], 2).sum(dim=1)
            c2 = torch.pow(outer, 2).sum(dim=1)
            diou = area_i / area_u - r2 / c2
            order = order[diou < nms_thresh]
        return keep

    def forward(self, predict, conf_thresh=0.4, nms_thresh=0.6):
        output_ = []
        for output in predict:
            batch = output.shape[0]
            size = output.shape[2]
            output = output.reshape(self.batch, 3, 85, size, size).permute(0, 3, 4, 1, 2)
            output[:, :, :, :, 0:2] = torch.sigmoid(output[:, :, :, :, 0:2]) * 1.05 - 0.025 + self.grid[size]
            output[:, :, :, :, 0:2] *= 608 // size
            output[:, :, :, :, 2:4] = torch.exp(output[:, :, :, :, 2:4]) * self.anchor[size]
            output[:, :, :, :, 4: ] = torch.sigmoid(output[:, :, :, :, 4: ])
            output = output.reshape(batch, size*size*3, 85)
            output_.append(output)
        output_ = torch.cat(output_, dim=1)
        output_[:, :, 5:] *= output_[:, :, 4:5].repeat(1, 1, 80)
        detection = []
        for b in range(self.batch):
            boxes = []
            for c in range(80):
                keep = self.NMS(output_[b, :, :4], output_[b, :, c+5], conf_thresh, nms_thresh)
                boxes.append(torch.cat([torch.full([len(keep), 1], c).to(self.device), output_[b, keep, :4], output_[b, keep, c+5:c+6]], dim=1))
            detection.append(torch.cat(boxes, dim=0))                
        return detection


class Yolo_Loss(nn.Module):
    def __init__(self, batch, device):
        super().__init__()
        self.batch = batch
        self.device = device
        self.grid = {}
        self.anchor = {}
        self.reference = {}
        anchor_dict = {
            76 : [[28, 28], [46, 45], [64, 66]] ,
            38 : [[102, 74], [78, 115], [132, 113]] ,
            19 : [[149, 163], [174, 268], [257, 176]]
        }
        for size in anchor_dict.keys():
            grid_x = torch.linspace(0, size-1, size).view(1, 1, size, 1, 1).repeat(batch, size, 1, 3, 1)
            grid_y = torch.linspace(0, size-1, size).view(1, size, 1, 1, 1).repeat(batch, 1, size, 3, 1)
            grid = torch.cat([grid_x, grid_y], dim=4)
            anchor_w = [torch.full([batch, size, size, 1, 1], anchor_dict[size][i][0]) for i in range(3)]
            anchor_w = torch.cat(anchor_w, dim=3)
            anchor_h = [torch.full([batch, size, size, 1, 1], anchor_dict[size][i][1]) for i in range(3)]
            anchor_h = torch.cat(anchor_h, dim=3)
            anchor = torch.cat([anchor_w, anchor_h], dim=4)
            reference = torch.tensor(anchor_dict[size])
            self.grid[size] = (grid.to(self.device))
            self.anchor[size] = (anchor.to(self.device))
            self.reference[size] = (reference.to(self.device))

    def IoU(self, boxes_a, boxes_b, align=False, cross=True, DIoU=False):
        if align == True:
            boxes_a = torch.cat([boxes_a / 2, boxes_a], dim=1)
            boxes_b = torch.cat([boxes_b / 2, boxes_b], dim=1)
        if cross == True:
            boxes_a = boxes_a.view(boxes_a.shape[0], 1, 4).repeat(1, boxes_b.shape[0], 1)
            boxes_b = boxes_b.view(1, boxes_b.shape[0], 4).repeat(boxes_a.shape[0], 1, 1)
        Boxes_a = torch.cat([boxes_a[..., :2] - boxes_a[..., 2:] / 2, boxes_a[..., :2] + boxes_a[..., 2:] / 2], dim=-1)
        Boxes_b = torch.cat([boxes_b[..., :2] - boxes_b[..., 2:] / 2, boxes_b[..., :2] + boxes_b[..., 2:] / 2], dim=-1)
        inter_1 = torch.maximum(Boxes_a[..., :2], Boxes_b[..., :2])
        inter_2 = torch.minimum(Boxes_a[..., 2:], Boxes_b[..., 2:])
        inter = torch.clamp(inter_2 - inter_1, min=0)
        area_a = boxes_a[..., 2] * boxes_a[..., 3]
        area_b = boxes_b[..., 2] * boxes_b[..., 3]
        area_i = inter[..., 0] * inter[..., 1]
        area_u = area_a + area_b - area_i
        iou = area_i / (area_u + 1e-7)
        if DIoU == False:
            return iou
        outer_1 = torch.minimum(Boxes_a[..., :2], Boxes_b[..., :2])
        outer_2 = torch.maximum(Boxes_a[..., 2:], Boxes_b[..., 2:])
        outer = torch.clamp(outer_2 - outer_1, min=0)
        r2 = torch.pow(boxes_a[..., :2] - boxes_b[..., :2], 2).sum(dim=-1)
        c2 = torch.pow(outer, 2).sum(dim=-1)
        diou = iou - r2 / (c2 + 1e-7)
        return diou

    def forward(self, predict, label, pos_thresh=0.2, neg_thresh=0.7):
        loss_box, loss_obj, loss_cls = 0, 0, 0
        for output in predict:
            batch = output.shape[0]
            size = output.shape[2]
            output = output.reshape(batch, 3, 85, size, size).permute(0, 3, 4, 1, 2) 
            output[:, :, :, :, 0:2] = torch.sigmoid(output[:, :, :, :, 0:2]) * 1.05 - 0.025 + self.grid[size]
            output[:, :, :, :, 0:2] *= 608 // size
            output[:, :, :, :, 2:4] = torch.exp(output[:, :, :, :, 2:4]) * self.anchor[size]
            output[:, :, :, :, 4: ] = torch.sigmoid(output[:, :, :, :, 4: ])
            target = torch.zeros(batch, size, size, 3, 85).to(self.device)
            pos_mask = torch.zeros(batch, size, size, 3, dtype=torch.bool).to(self.device)
            neg_mask = torch.zeros(batch, size, size, 3, dtype=torch.bool).to(self.device)
            for b in range(self.batch):
                truth = label[b, :(label[b].sum(dim=1) > 0).sum(dim=0), :]
                pos_iou = self.IoU(truth[:, 3:], self.reference[size], align=True)
                for t in range(truth.shape[0]):
                    i = torch.div(truth[t, 0], 608//size, rounding_mode="floor").to(torch.int64)
                    j = torch.div(truth[t, 1], 608//size, rounding_mode="floor").to(torch.int64)
                    a = ((pos_iou[t, :] > pos_thresh).nonzero())[:, 0]
                    if a.shape[0] == 0: continue
                    pos_mask[b, j, i, a] = True
                    target[b, j, i, a, :4] = truth[t, 1:].view(1, 4).repeat(a.shape[0], 1)
                    target[b, j, i, a, 4] = torch.ones(a.shape[0]).to(self.device)
                    target[b, j, i, a, (5+truth[t, 0]).to(torch.int64)] = torch.ones(a.shape[0]).to(self.device)
                neg_iou = self.IoU(output[b].reshape(size*size*3, 85)[:, :4], truth[:, 1:])
                neg_ind = (torch.max(neg_iou, 1)[0] < neg_thresh).view(size, size, 3)
                neg_mask[b, neg_ind] = True
            if pos_mask.sum() == 0 or neg_mask.sum() == 0: continue
            loss_box += (1 - self.IoU(output[pos_mask][:, :4], target[pos_mask][:, :4], cross=False, DIoU=True)).mean() / batch
            loss_obj += nn.functional.binary_cross_entropy(output[pos_mask][:, 4], target[pos_mask][:, 4]) / batch
            loss_obj += nn.functional.binary_cross_entropy(output[neg_mask][:, 4], target[neg_mask][:, 4]) / batch
            loss_cls += nn.functional.binary_cross_entropy(output[pos_mask][:, 5:], target[pos_mask][:, 5:]) / batch
        loss = loss_box + loss_obj + loss_cls
        return loss, loss_box, loss_obj, loss_cls


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Yolo_Detector(batch=16, device=device)
    input_ = [((torch.rand(16, 255, s, s)-0.5)*1.09).to(device) for s in [76, 38, 19]]
    output_ = detector(input_)
    
    import random
    from dataset import Yolo_Dataset
    from tools import draw_boxes, show_image
    test_dataset =  Yolo_Dataset("train_raw")
    for i in range(16):
        img, lbl = test_dataset.__getitem__(random.randrange(0, 11000))

        print(img.shape)

        img = draw_boxes(img, output_[i])
        show_image(img)     

    '''
    criterion = Yolo_Loss()
    boxes_a = torch.tensor([[100., 100., 100., 100.]])
    boxes_b = torch.tensor([[150., 150., 100., 100.]])
    print(criterion.IoU(boxes_a, boxes_b, CIoU=True))
    '''
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Loss = Yolo_Loss(16, device)
    predict = [(torch.rand(16, 255, s, s)-0.5).to(device) for s in [76, 38, 19]]
    label = [torch.randint(0, 80, (16, 4, 1)), torch.randint(0, 608, (16, 4, 2)), torch.randint(100, 150, (16, 4, 2))]
    label = torch.cat(label, dim=2).to(dtype=torch.float, device=device)
    loss, loss_box, loss_obj, loss_cls= Loss(predict, label)
    print(loss, loss_box, loss_obj, loss_cls)
    
    '''

