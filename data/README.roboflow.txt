
Atlantic_Salmon_ROIs - v7 2025-01-16 3:15pm
==============================

This dataset was exported via roboflow.com on January 16, 2025 at 2:16 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 808 images.
Roi are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 1080x350 (Stretch)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 5 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random shear of between -3° to +3° horizontally and -3° to +3° vertically
* Random brigthness adjustment of between 0 and +4 percent
* Random exposure adjustment of between -5 and +5 percent
* Random Gaussian blur of between 0 and 0.5 pixels


