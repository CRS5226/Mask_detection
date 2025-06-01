# Face Mask Detection using Deep Learning

## Overview

This project provides a real-time face mask detection system using deep learning and computer vision. It detects faces in images or video streams and classifies whether each detected face is wearing a mask or not. The system is built using TensorFlow/Keras with a MobileNetV2 backbone for efficient and accurate mask detection.

## Key Features

- **Real-time Detection:** Detects faces and classifies mask usage in live video streams or video files.
- **Pre-trained Model Included:** Use the provided trained model for immediate testing—no need to retrain.
- **Alert System:** Notifies (sound, desktop notification, email, screenshot) when someone is not wearing a mask.
- **Easy Training:** Includes scripts to train your own mask detector on custom datasets.
- **Visualization:** Plots training accuracy/loss and provides sample output images.
- **Modular Code:** Well-organized scripts for training, testing, and video inference.

## Project Structure

```
Mask_detection/
│
├── detect_mask_video.py         # Real-time mask detection from webcam/video
├── mask_test.py                 # Test mask detection on webcam/video
├── train_mask_detector.py       # Script to train or test the mask detector model
├── mask_detector.model          # Trained Keras model (ready to use)
├── requirements.txt             # Python dependencies
├── plot.png                     # Training accuracy/loss plot
├── log.txt                      # Log of detection events
├── records.txt                  # Additional records
├── maskvideo.mp4                # Sample video for testing
├── README.md                    # Project documentation (this file)
│
├── dataset/                     # Training dataset
│   ├── with_mask/               # Images with masks
│   └── without_mask/            # Images without masks
│
├── face_detector/               # Face detection model files
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── MobileNetSSD_deploy.caffemodel
│   └── MobileNetSSD_deploy.prototxt.txt
```

## How to Run

### 1. Install Requirements

Open a terminal in the `Mask_detection` folder and run:

```
pip install -r requirements.txt
```

### 2. Use the Pre-trained Model for Testing

You can directly use the provided `mask_detector.model` for testing without retraining.

#### To test on a video file:

```
python train_mask_detector.py --video maskvideo.mp4
```

- Replace `maskvideo.mp4` with your own video file if desired.
- The script will use the trained model to detect masks in the video.

#### To test on webcam:

```
python detect_mask_video.py
```

### 3. (Optional) Train the Mask Detector

If you want to retrain the model with your own data:

```
python train_mask_detector.py
```

- Place your images in `dataset/with_mask/` and `dataset/without_mask/`.

### 4. Alerts and Logging

- **Sound:** Beeps when a person without a mask is detected.
- **Notification:** Desktop notification appears for no-mask cases.
- **Screenshot:** Takes a screenshot when a violation is detected.
- **Email:** Sends an email alert (configure sender email and password in the script).
- **Log:** Events are logged in `log.txt`.

## Run on Google Colab or Kaggle (GPU Recommended)

- Upload the code and dataset to [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code).
- Enable GPU acceleration for faster training and inference.
- Install dependencies using `!pip install -r requirements.txt`.
- Adjust file paths as needed for the Colab/Kaggle environment.

## Advantages

- **Fast and Accurate:** Uses MobileNetV2 for efficient inference on CPU and GPU.
- **Ready to Use:** Pre-trained model included for immediate testing.
- **Customizable:** Easily retrainable on new datasets or for different mask types.
- **Comprehensive Alerts:** Multiple notification methods for safety compliance.
- **Open Source:** Code is modular and easy to extend.

## Important Links

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Keras Documentation](https://keras.io/)

---

**For questions or contributions, please open an issue or pull request. Stay safe!**
