import torch
import numpy as np
import pandas as pd


class YOLO_Evaluator():
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.detections = []
        self.labels = []
        self.corrects = []
        self.confusion = np.zeros([len(classes)+1, len(classes)+1])
        self.counts = {"TP": 0, "FP": 0, "FN": 0}
    
    def IoU(self, boxes_a, boxes_b):
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
        area_u = area_a + area_b - area_i + 1e-9
        iou = area_i / area_u
        return iou

    def process(self, detection, label):
        self.detections.append(detection)
        self.labels.append(label)
        nd, nl = detection.shape[0], label.shape[0]
        correct = torch.zeros(nd, 10)
        iou = self.IoU(label[:, 1:], detection[:, :4])
        class_match = label[:, 0].view(nl, 1).repeat(1, nd) == detection[:, 5].view(1, nd).repeat(nl, 1)
        for i, iouv in enumerate(np.arange(0.5, 1.0, 0.05)):
            x = torch.where((iou > iouv) & class_match)
            matches = torch.cat([torch.stack(x, 1), iou[x[0], x[1]][:, None]], 1)
            matches = matches[matches[:, 2].argsort(descending=True)]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort(descending=True)]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].long(), i] = True
        self.corrects.append(correct)

        detection = detection[detection[:, 4] > 0.25]
        detect_cls = detection[:, 5].int()
        label_cls = label[:, 0].int()
        iou = self.IoU(label[:, 1:], detection[:, :4])
        x = torch.where(iou > 0.45)
        matches = torch.cat([torch.stack(x, 1), iou[x[0], x[1]][:, None]], 1)
        matches = matches[matches[:, 2].argsort(descending=True)]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort(descending=True)]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        for i in range(matches.shape[0]):
            self.confusion[detect_cls[i, 1], detect_cls[i, 0]] += 1
            self.counts["TP"] += 1
        for i, dc in enumerate(detect_cls):
            if not any(detect_cls[:, 1] == i):
                self.confusion[dc, len(self.classes)] += 1
                self.counts["FP"] += 1
        for i, gc in enumerate(label_cls):
            if not any(label_cls[:, 0] == i):
                self.confusion[len(self.classes), gc] += 1
                self.counts["FN"] += 1

    def compute(self):
        detections = torch.cat(self.detections, 0).cpu().numpy()
        labels = torch.cat(self.labels, 0).cpu().numpy()
        corrects = torch.cat(self.corrects, 0).cpu().numpy()
        order = np.argsort(detections[:, 4])
        detections = detections[order]
        corrects = corrects[order]
        classes, class_cnt = np.unique(labels, return_counts=True)
        #conf_x = np.linspace(0, 1, 1000)
        #p, r, ap = np.zeros([classes.shape[0], 1000]), np.zeros([classes.shape[0], 1000]), np.zeros([classes.shape[0], 10])
        for ci, c in enumerate(classes):
            i = detections[:, 5] == c
            tpc = corrects[i].cumsum(0)
            fpc = (1 - corrects[i]).cumsum(0)
            recall = tpc / (class_cnt[ci] + 1e-16)
            precision = tpc / (tpc + fpc + 1e-16)



        self.result = []
        self.p_plot = []
        self.r_plot = []
        self.f1_plot = []
        self.ap_plot = []
        self.confusion_plot = []





