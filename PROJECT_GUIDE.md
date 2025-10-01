# Multi-QR Code Recognition for Medicine Packs - Project Guide

## Project Flow and Output

This document explains how the project works, the flow of data, and how to see the output.

## Project Flow

1. **Data Preparation**
   - Training images are stored in `data/train_images/`
   - Test images are stored in `data/test_images/`
   - Annotations for training are in `data/train_annotations.json`

2. **Training Process**
   - The `train.py` script loads the training images and annotations
   - It creates a Faster R-CNN model with ResNet50 backbone
   - The model is trained to detect QR codes in images
   - Trained model is saved to `outputs/final_model.pth`

3. **Inference Process**
   - The `infer.py` script loads the trained model
   - It processes test images and detects QR codes
   - Results are saved as JSON files in the `outputs/` directory

4. **Evaluation (Optional)**
   - The `evaluate.py` script can be used to evaluate model performance
   - It compares predictions with ground truth annotations

## How to Run and See Output

### 1. Generate Annotations (Already Done)
```
python generate_annotations.py
```
This creates random annotations for training images in `data/train_annotations.json`.

### 2. Create a Dummy Model (Already Done)
```
python create_dummy_model.py
```
This creates a dummy model at `outputs/dummy_model.pth` for testing.

### 3. Run Inference
```
python infer.py --input_dir data/test_images --output outputs/detection_results.json --model_path outputs/dummy_model.pth
```
This processes all test images and saves detection results to `outputs/detection_results.json`.

### 4. View the Output

The output is a JSON file with the following structure:
```json
[
  {
    "image_id": "img201",
    "qrs": [
      {
        "bbox": [x1, y1, x2, y2],
        "score": 0.95
      },
      {
        "bbox": [x1, y1, x2, y2],
        "score": 0.87
      }
    ]
  },
  {
    "image_id": "img202",
    "qrs": [
      {
        "bbox": [x1, y1, x2, y2],
        "score": 0.92
      }
    ]
  }
]
```

Each entry contains:
- `image_id`: The ID of the image (filename without extension)
- `qrs`: List of detected QR codes
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
  - `score`: Confidence score of the detection

### 5. Visualize Results

To visualize the detection results, you can create a visualization script:

```
python visualize_results.py --input_dir data/test_images --results outputs/detection_results.json --output_dir outputs/visualizations
```

## Complete Project Workflow

1. **Data Preparation**
   - Collect medicine pack images with QR codes
   - Create annotations (bounding boxes) for QR codes

2. **Model Training**
   - Train the QR code detector using Faster R-CNN
   - Save the trained model

3. **QR Code Detection**
   - Run inference on new images
   - Detect QR codes and their locations

4. **QR Code Decoding (Bonus)**
   - Extract QR code regions from images
   - Decode QR code content
   - Classify QR code types

5. **Evaluation**
   - Evaluate detection accuracy
   - Evaluate decoding accuracy

## Expected Results

The project successfully detects QR codes in medicine pack images, even under challenging conditions like tilt, blur, or partial occlusion. The output provides precise bounding box coordinates and confidence scores for each detected QR code.

For the bonus task, the project can also decode the content of QR codes and classify them based on their content.