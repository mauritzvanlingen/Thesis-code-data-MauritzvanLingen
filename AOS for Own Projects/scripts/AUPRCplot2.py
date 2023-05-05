import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc




# Define ground truth and detection boxes and scores for multiple images


image_data = [    {        'gt_boxes': [[346,118,390,164],[124,114,159,194],[288,144,159,194]],
        'gt_labels': [1, 1, 1],
        'det_boxes': [[328,117,399,170],[280,185,376,233]],
        'det_scores': [0.182, 0.103],
        'det_labels': [1,1]
    },
    {
        'gt_boxes': [[468, 256, 547, 256],[263, 266, 349, 318],[460, 342, 545, 398] ],
        'gt_labels': [1, 1, 1],
        'det_boxes': [[468, 256, 547, 256],[263, 266, 349, 318],[460, 342, 545, 398],[456,671,556,671],[612,692,716,692]],
        'det_scores': [0.431, 0.388, 0.432, 0.114, 0.12],
        'det_labels': [1, 1, 1, 1, 1]
    },
    {
        'gt_boxes': [[343,293,430,350],[126,300,242,370],[345,392,428,457]],
        'gt_labels': [1, 1, 1],
        'det_boxes': [[343,293,430,350],[126,298,242,373],[345,392,428,457]],
        'det_scores': [0.313, 0.01, 0.296],
        'det_labels': [1, 1, 0, 1]
    }
]
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def calculate_precision_recall(image_data, class_id):
    tp = 0
    fp = 0
    fn = 0
    for image in image_data:
        gt_boxes = image['gt_boxes']
        gt_labels = image['gt_labels']
        det_boxes = image['det_boxes']
        det_scores = image['det_scores']
        det_labels = image['det_labels']
        for i, det_box in enumerate(det_boxes):
            if det_labels[i] == class_id:
                max_iou = 0
                for j, gt_box in enumerate(gt_boxes):
                    if gt_labels[j] == class_id:
                        iou = calculate_iou(det_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                if max_iou >= 0.5:
                    tp += 1
                else:
                    fp += 1
        for j, gt_box in enumerate(gt_boxes):
            if gt_labels[j] == class_id:
                found_match = False
                for i, det_box in enumerate(det_boxes):
                    if det_labels[i] == class_id:
                        iou = calculate_iou(det_box, gt_box)
                        if iou >= 0.5:
                            found_match = True
                            break
                if not found_match:
                    fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

print(calculate_precision_recall(image_data=image_data, class_id= 1))