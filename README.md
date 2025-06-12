Deep Learning-Based Cardiac MRI Timelapse Analysis for Early Disease Detection
Cardiovascular diseases are a global health challenge. This project harnesses the power of deep learning to analyze cardiac cine MRI timelapse sequences, aiming for the early detection of heart conditions. We provide an AI-driven system that helps doctors quickly understand heart structure, analyze its motion, spot anomalies, and even predict potential disease progression.

Features
Cardiac Structure Segmentation: Our AI uses a U-Net model to precisely outline heart chambers (like the Left Ventricle, Right Ventricle, and Myocardium) on MRI slices.
Temporal Dynamics Analysis: The system tracks how these heart structures move and change across the entire MRI video sequence, providing insights into cardiac function.
Anomaly Detection: It identifies abnormal or irregular cardiac motion patterns that could indicate early disease.
Disease Progression Prediction: Based on the observed heart dynamics, the system can offer a projection of future cardiomyopathy risk.
Interactive Visualizations: We've built an intuitive Streamlit web interface for real-time exploration of segmentation results, heart metrics, and risk predictions.
User-Friendly Interface: A web-based application allows medical professionals to easily upload MRI sequences and get instant, visual analyses.
How It Works (The Magic Behind the Scenes)
This project involves several key steps, from preparing raw MRI data to delivering insights via a user-friendly interface:

Data Preparation (ACDC_processing.py):

We start with raw cardiac MRI data (from datasets like ACDC).
This script takes complex .nii MRI files and converts them into individual, standardized image slices (.png format) ready for AI processing.
It also normalizes images and scales masks to ensure consistency for the deep learning model.
Dataset Handling (mri_dataset.py):

This script acts as the data manager for our AI.
It loads the processed MRI image slices and their corresponding "masks" (which show where the heart structures are) into a format that our deep learning model can understand and learn from.
The Brain - Our AI Model (Model.py):

This is where the deep learning magic happens!
It defines a U-Net neural network, which is excellent at image segmentation (i.e., outlining specific structures in images).
The script also includes the training process: it teaches the U-Net to accurately identify heart chambers by showing it thousands of MRI images and their correct outlines.
It uses a "Dice Loss" function to evaluate how well the model is performing, pushing it to get more and more accurate.
The User Interface (Interface.py - Our Streamlit App):

This is what you see and interact with!
It loads the pre-trained AI model (the "brain").
You can upload or select folders containing cardiac MRI sequences (healthy, diseased, or a new test case).
The app then sends these images to our AI model for analysis.
It generates real-time segmentations, visualizes heart motion, compares it to healthy and diseased patterns, diagnoses potential issues (like cardiomyopathy), and even forecasts future risk, all presented interactively in your web browser.

Structure of project
.
├── ACDC_processing.py      # Script to preprocess raw ACDC MRI data into images and masks
├── Model.py                # Defines the U-Net deep learning model and training logic
├── mri_dataset.py          # Handles loading MRI image and mask data for the model
└── Interface.py            # The Streamlit web application for analysis and visualization
└── unet_epoch_5.pth        # (Example) A pre-trained model checkpoint
└── dataset/
    ├── train/
    │   ├── images/         # Processed training MRI images
    │   └── masks/          # Corresponding training segmentation masks
    └── test/
        ├── images/         # Processed testing MRI images
        └── masks/          # Corresponding testing segmentation masks
└── Database/               # Original ACDC dataset (expected if you run ACDC_processing)
    ├── training/
    └── testing/
└── healthy/
    └── images/             # Sample healthy patient images for demo
└── unhealthy/
    └── images/             # Sample unhealthy patient images for demoFlowchart Description: Cardiac MRI Analysis Pipeline
Imagine a conveyor belt where raw MRI data enters on one side, and insightful diagnostic information comes out the other!

Start with Raw MRI Data: The process begins with complex .nii files from the ACDC dataset, representing full 3D cardiac MRI scans.
Data Preparation (ACDC_processing.py):
These raw files are fed into the ACDC_processing.py script.
This script slices the 3D data into individual 2D images and their corresponding "ground truth" (correct) masks.
It then resizes, normalizes, and saves them as .png files into structured dataset/train/images, dataset/train/masks, dataset/test/images, and dataset/test/masks folders.
Dataset Loading (mri_dataset.py):
During model training, the mri_dataset.py script acts as a bridge. It reads these processed .png image and mask files.
It then organizes them into "batches" that the deep learning model can efficiently process.
Model Training (Model.py):
The core Model.py script takes the prepared data batches.
It uses the U-Net deep learning architecture to learn how to identify and outline heart structures from the images.
Through an iterative training process (epochs), the model adjusts its internal parameters to minimize the dice_loss, becoming more accurate at segmentation.
Periodically, or at the end, a trained model (.pth file) is saved.
User Interaction (Interface.py - Streamlit App):
This is the front-end for users.
It loads the trained unet_epoch_5.pth model.
Users can select folders containing MRI sequences (e.g., test case, healthy, diseased examples).
The app feeds individual frames from these sequences to the loaded model for real-time segmentation.
It then calculates metrics like segmentation area, analyzes motion variation, and uses these to provide a diagnosis and project future risk.
All these results are presented dynamically through interactive plots, images, and text within the Streamlit web interface.
Insights and Diagnosis: The end result is a comprehensive analysis of the cardiac MRI timelapse, helping medical professionals in early disease detection and assessment.


         ┌─────────────────────┐
         │  ACDC MRI Dataset   │
         └─────────┬───────────┘
                   ▼
        ┌────────────────────────┐
        │  image_converter.py    │
        │ → PNG slices/masks     │
        └─────────┬──────────────┘
                  ▼
        ┌────────────────────────┐
        │     train.py           │
        │ → U-Net segmentation   │
        └─────────┬──────────────┘
                  ▼
        ┌────────────────────────┐
        │    evaluate.py         │
        │ → Dice, pixel accuracy │
        └─────────┬──────────────┘
                  ▼
        ┌───────────────────────────────┐
        │ temporal_analysis.py          │
        │ → Frame-wise LV/RV/MYO curves │
        └─────────┬─────────────────────┘
                  ▼
        ┌──────────────────────────────┐
        │  anomaly_detection.py        │
        │ → Detect motion deviations   │
        └─────────┬────────────────────┘
                  ▼
        ┌──────────────────────────────┐
        │      visualize.py            │
        │ → Graphs, overlays           │
        └─────────┬────────────────────┘
                  ▼
        ┌──────────────────────────────┐
        │     app.py (Streamlit App)   │
        │ → Frame slider, anomaly map  │
        └──────────────────────────────┘
