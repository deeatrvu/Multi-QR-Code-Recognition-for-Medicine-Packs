# Multi-QR Code Recognition for Medicine Packs

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning-based system for detecting and decoding multiple QR codes on medicine packages using Faster R-CNN with ResNet50 backbone.

## Features

- **Multi-QR Code Detection**: Accurately detects multiple QR codes in a single image
- **Robust Recognition**: Works under challenging conditions (tilt, blur, partial occlusion)
- **QR Code Decoding**: Extracts and decodes content from detected QR codes
- **Medicine Pack Specific**: Optimized for pharmaceutical packaging

## Project Structure

```
├── data/
│   ├── train_images/      # Training images
│   ├── test_images/       # Test images
│   └── train_annotations.json  # Annotations for training
├── src/
│   ├── datasets/          # Dataset loading and processing
│   ├── models/            # Model architecture
│   └── utils/             # Utility functions
├── outputs/               # Saved models and results
├── train.py               # Training script
├── infer.py               # Inference script
├── evaluate.py            # Evaluation script
└── generate_annotations.py # Annotation generation script
```

## Installation

```bash
# Clone the repository
git clone https://github.com/deeatrvu/Multi-QR-Code-Recognition-for-Medicine-Packs.git
cd Multi-QR-Code-Recognition-for-Medicine-Packs

# Install dependencies
pip install -r requirements.txt

# Optional: Install ZBar for QR code decoding
# Windows: Download from http://zbar.sourceforge.net/
# Linux: sudo apt-get install libzbar0
```

## Usage

### 1. Generate Annotations
```bash
python generate_annotations.py
```

### 2. Train the Model
```bash
python train.py --data_dir data/train_images --annotations data/train_annotations.json --output_dir outputs --epochs 10
```

### 3. Run Inference
```bash
python infer.py --input_dir data/test_images --output outputs/detection_results.json --model_path outputs/final_model.pth
```

### 4. Evaluate Results
```bash
python evaluate.py --predictions outputs/detection_results.json --ground_truth data/test_annotations.json
```

## Results

The model achieves:
- High precision and recall in QR code detection
- Robust performance under various lighting conditions
- Accurate decoding of QR code content

## Future Work

- Implement real-time detection for video streams
- Add support for additional barcode types
- Optimize for mobile deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Torchvision for the pre-trained models
- ZBar library for QR code decoding