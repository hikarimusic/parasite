import cv2
import numpy as np
import colorsys
from tqdm import tqdm


def draw_boxes(image, boxes):
    image = image.cpu().detach().numpy()
    image = image.transpose(1, 2, 0)
    boxes_n = (boxes.sum(dim=1) > 0).sum(dim=0)
    boxes = boxes[:boxes_n].tolist()
    for box in boxes:
        box = [box[0], box[1]-box[3]/2, box[2]-box[4]/2, box[1]+box[3]/2, box[2]+box[4]/2]
        box = [int(c) for c in box]
        color = colorsys.hsv_to_rgb(box[0]/11, 1.0, 1.0)
        cv2.rectangle(image, (box[1], box[2]), (box[3], box[4]), color[::-1], 2)
    return image


def show_image(image, name="Image"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 900, 900) 
    cv2.imshow(name, image)
    cv2.waitKey(0)   


def calc_anchors():
    from sklearn.cluster import KMeans
    from dataset import Yolo_Dataset
    dataset = Yolo_Dataset("train_raw")
    boxes = []
    for i in tqdm(range(dataset.__len__())):
        _ , label = dataset.__getitem__(i)
        boxes_n = (label.sum(dim=1) > 0 ).sum(dim=0)
        boxes += label[:boxes_n, 3:].tolist()
    kmeans = KMeans(n_clusters=9).fit(boxes)
    anchors = np.rint(kmeans.cluster_centers_).astype(int).tolist()    
    anchors = sorted(anchors, key=lambda x: min(x[0], x[1]))
    print(anchors) # [[28, 28], [46, 45], [64, 66], [102, 74], [78, 115], [132, 113], [149, 163], [174, 268], [257, 176]]