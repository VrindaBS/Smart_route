# Smart Route – Real-Time AI-Based Approach for Congestion Control

Smart Route is an AI-powered system that uses computer vision to detect traffic congestion in real-time using live CCTV feeds. The system helps optimize urban mobility by classifying congestion levels and suggesting alternative routes dynamically.

## Features
- Real-time video frame analysis from live CCTV feeds
- Congestion classification into five levels
- Pretrained convolutional neural network for efficient processing
- Route optimization suggestions based on congestion levels

## Tech Stack
- Python, OpenCV, TensorFlow/Keras
- Video dataset for training and evaluation
- Flask (optional) for deployment interface

## How It Works
1. Traffic video datasets were used to extract frames.
2. Congestion levels were manually labeled across five categories.
3. A convolutional model was trained to classify traffic levels.
4. The trained model was then used on real-time feeds to predict congestion.
5. Based on predictions, less congested routes are recommended.

## Status
Submitted to Yukti Innovation Challenge 2025  
**Research Paper Link:** [Smart Route Paper](https://drive.google.com/file/d/1QMw4U-e4-dIgpFjFJapB2xaaUkldzEV_/view?usp=sharing)

## Author
Vrinda B S
