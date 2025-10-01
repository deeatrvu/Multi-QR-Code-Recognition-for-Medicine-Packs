import os
import json
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from src.models.detector import QRCodeDetector

# Import decoder conditionally to handle missing libzbar dependency
decode_qr_codes = None

def run_inference(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import decoder only if needed
    global decode_qr_codes
    if args.decode:
        try:
            from src.utils.decoder import decode_qr_codes
            print("QR code decoding enabled")
        except ImportError:
            print("Warning: QR code decoding libraries not available. Running without decoding.")
            args.decode = False
    
    # Load model
    model = QRCodeDetector(num_classes=2)
    model.load(args.model_path, device)
    model.to(device)
    model.eval()
    
    # Get all image files
    image_files = []
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_id = os.path.splitext(filename)[0]
            image_files.append((image_id, filename))
    
    # Run inference
    results = []
    for image_id, filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.input_dir, filename)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model([image_tensor])
        
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence
        keep = scores > args.confidence_threshold
        boxes = boxes[keep].tolist()
        
        # Create result
        result = {
            "image_id": image_id,
            "qrs": [{"bbox": box} for box in boxes]
        }
        
        # Decode QR codes if requested and available
        if args.decode and decode_qr_codes:
            try:
                qr_values = decode_qr_codes(image, boxes)
                for i, value in enumerate(qr_values):
                    if value:
                        result["qrs"][i]["value"] = value
            except Exception as e:
                print(f"Error decoding QR code: {e}")
        
        results.append(result)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference completed! Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with QR code detector")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--decode', action='store_true', help='Decode QR codes (for bonus task)')
    
    args = parser.parse_args()
    run_inference(args)