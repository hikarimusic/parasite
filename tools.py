import cv2
import torch
import colorsys


def draw_boxes(image, boxes, tensor=False):
    if tensor == True:
        image = image.cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        boxes = boxes.tolist()
    for box in boxes:
        box = [box[0], box[1]-box[3]/2, box[2]-box[4]/2, box[1]+box[3]/2, box[2]+box[4]/2]
        box = [int(c) for c in box]
        color = colorsys.hsv_to_rgb(box[0]/11, 1.0, 1.0)
        cv2.rectangle(image, (box[1], box[2]), (box[3], box[4]), color[::-1], 1)
    return image


def show_image(image, name="Image"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 900, 900) 
    cv2.imshow(name, image)
    cv2.waitKey(0)   