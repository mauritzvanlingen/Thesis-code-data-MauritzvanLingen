import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from data_detections import TAOS_data_original, TAOS_data_Gaussian_filtering, TAOS_data_CLAHE, TAOS_data_AHE

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1) 
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        union_area = 0.00001    # to make sure we dont get devide by 0
    return intersection_area / union_area
# [x_min, y_min, x_max, y_max] for the GT and detection boxes


def calculate_prc(image_data, class_id):
    precisions = []
    recalls = []
    for threshold in np.arange(0.05, 1.0, 0.05):
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
                
                if det_labels[i] == class_id and det_scores[i] >= threshold:
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
                        if det_labels[i] == class_id and det_scores[i] >= threshold:
                            iou = calculate_iou(det_box, gt_box)
                            if iou >= 0.5:
                                found_match = True
                                break
                    if not found_match:
                        fn += 1
    
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        if threshold == 0.05:
            print(f"precision: {precision}, recall: {recall}")
            
            print(f"TP: {tp}, FP: {fp}, FN: {fn}\n")
        
        precisions.append(precision)
        recalls.append(recall)
       
    return precisions, recalls

def plot_prc(image_data1, image_data2, image_data3, image_data4, class_id):
    
    plt.figure(figsize=(8, 6)) 
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    lines = []  # To store the plotted lines
    labels = []  # To store the labels for each line
    
    for i, data in enumerate([image_data1, image_data2, image_data3, image_data4]):
        precision, recall = calculate_prc(data, class_id)
        data_names = ["Original TAOS", "Gaussian filtering TAOS", "CLAHE TAOS", "AHE TAOS"]
        # Calculate AUPRC
        precision = np.array(precision)
        recall = np.array(recall)

        sorted_indices = np.argsort(recall)
        recall = recall[sorted_indices]
        precision = precision[sorted_indices]
        auprc = np.trapz(precision, recall)
        
        # Plot the precision-recall curve and store the line and label
        line, = plt.plot(recall, precision, linestyle='-', linewidth=2)
        lines.append(line)
        labels.append(f' {data_names[i]}, AUPRC = {auprc:.4f}')
        
    plt.legend(lines, labels)  # Add legend with the stored lines and labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Class {class_id} (persons))')
    plt.show()

#plot_prc(TAOS_data_original, 1)
#plot_prc(TAOS_data_Gaussian_filtering, 1)
plot_prc(TAOS_data_original, TAOS_data_Gaussian_filtering, TAOS_data_CLAHE, TAOS_data_AHE, 1)

