# Thesis code B,Sc, thesis Mauritz van Lingen (u791472)

This is the code and data produced by myself for this thesis.

The AOS software and F0 dataset used for this thesis can be found on (https://github.com/JKU-ICG/AOS). the C++ implementation of the LFR algorithm has been used to compute the integral images used fot his thesis. 

in the DET folder, the YOLO v3 implementation of the CNN used for object detection of the contrast (un)adjusted TAOS images that were generated using the LFR algorithm and F0 dataset.

## Modules

-      (C++ and Python code): computes integral images.
- [DET](DET/README.md)      (Python code): contains the person classification.
- [CAM](CAM/README.md)      (Python code): the module for triggering, recording, and processing thermal images.
- [PLAN](PLAN/README.md)    (Python code): implementation of our path planning and adaptive sampling technique.
- [DRONE](DRONE/README.md)  (C and Python code): contains the implementation for drone communication and the logic to perform AOS flights.
- [SERV](SERV/README.md)  (Rust code): contains the implementation of a dabase server to which AOS flights data are uploaded.

