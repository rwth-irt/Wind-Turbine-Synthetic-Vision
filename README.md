# Wind Turbine Synthetic Vision

## Description
This repository contains the source code used to generate synthetic drone imagery and to train a YOLOv11 object detection model based on the generated dataset.

The pipeline relies on BlenderProc for physically-based image synthesis, and parts of the code are adapted from the open-source YOLOv11 framework.

## Project Status
This is the camera-ready version of the code accompanying the ICMV 2025 paper:

**“Wind Turbine Feature Detection Using Deep Learning and Synthetic Data”**  
*Arash Shahirpour, Jakob Gebler, Manuel Sanders, Tim Reuscher*

Future development will include:
- Integration into a Docker environment
- Simplified parameterization and configurability
- A production-grade release

## Dependencies
- [BlenderProc](https://github.com/DLR-RM/BlenderProc) (GPLv3)
- [YOLOv11](link-to-the-fork-or-original-if-available) (AGPL-3.0)
- NumPy, Pillow, OpenCV, and other standard Python libraries

## Authors and Acknowledgment
- Arash Shahirpour  
- Jakob Gebler  
- Manuel Sanders  

With contributions and supervision by Tim Reuscher,  
Institute of Automatic Control – RWTH Aachen University

## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

It includes and modifies components from YOLOv11 (AGPL-3.0), and uses BlenderProc (GPLv3), both of which enforce free software licensing.  
See the [`LICENSE`](./LICENSE) file for full terms.

© 2025 Arash Shahirpour, Jakob Gebler, Manuel Sanders, RWTH Aachen University.
