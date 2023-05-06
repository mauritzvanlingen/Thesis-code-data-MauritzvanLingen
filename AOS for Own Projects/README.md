# Thesis code B,Sc, thesis Mauritz van Lingen (u791472)

This is the code and data produced by myself for this thesis.

The AOS software and F0 dataset used for this thesis can be found on (https://github.com/JKU-ICG/AOS). The C++ implementation of the LFR algorithm has been used to compute the integral images used fot his thesis. 

in the DET folder, the YOLO v3 implementation of the CNN used for object detection of the contrast (un)adjusted TAOS images that were generated using the LFR algorithm and F0 dataset.

## Results thesis

- contains the 19 TAOS (un)adjusted for contrast images
- the imgage and detections folder contains the contrast (un)adjusted TAOS images with detection boxed made by the YOLO v3 CNN.
- contains figure outputted by AUPRC_function.py

## Scripts
- AUPRC_function.py plots the PRC plot and calculates the respective AUPRC, the data_detections.py function as input.
- data_detections.py are lists with dictionaries. the dictionaries cntain info about the coordinates of the corners of the ground truth boxes and the classes. the      detection scores are also supplied with the confidence of the detections.
- context.py is a script that adjust the contrast of input images
- Sub_experiment.R contains the data of the sub experiment stored in a vector, the mean and standard deviations are calculated.
