import os
import json
import argparse
import numpy as np
from src.utils.utils import compute_iou

def evaluate_detection(pred_file, gt_file, iou_threshold=0.5):
    """
    Evaluate QR code detection performance
    
    Args:
        pred_file: Path to prediction JSON file
        gt_file: Path to ground truth JSON file
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        precision, recall, f1_score
    """
    # Load predictions and ground truth
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Convert ground truth to dictionary for easier lookup
    gt_dict = {item['image_id']: item['qrs'] for item in ground_truth}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred in predictions:
        image_id = pred['image_id']
        pred_boxes = [qr['bbox'] for qr in pred['qrs']]
        
        if image_id in gt_dict:
            gt_boxes = [qr['bbox'] for qr in gt_dict[image_id]]
            
            # Match predictions to ground truth
            matched_gt_indices = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt_indices:
                        continue
                    
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    # True positive
                    total_tp += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    # False positive
                    total_fp += 1
            
            # Count false negatives
            total_fn += len(gt_boxes) - len(matched_gt_indices)
        else:
            # All predictions are false positives if image_id not in ground truth
            total_fp += len(pred_boxes)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def evaluate_decoding(pred_file, gt_file):
    """
    Evaluate QR code decoding performance
    
    Args:
        pred_file: Path to prediction JSON file with decoded values
        gt_file: Path to ground truth JSON file with decoded values
        
    Returns:
        accuracy
    """
    # Load predictions and ground truth
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Convert ground truth to dictionary for easier lookup
    gt_dict = {}
    for item in ground_truth:
        image_id = item['image_id']
        gt_dict[image_id] = {}
        
        for qr in item['qrs']:
            if 'value' in qr:
                # Use bbox as key (convert to tuple for hashability)
                bbox_tuple = tuple(qr['bbox'])
                gt_dict[image_id][bbox_tuple] = qr['value']
    
    total_correct = 0
    total_predictions = 0
    
    for pred in predictions:
        image_id = pred['image_id']
        
        if image_id in gt_dict:
            for qr in pred['qrs']:
                if 'value' in qr:
                    total_predictions += 1
                    
                    # Find matching ground truth box
                    pred_box = qr['bbox']
                    best_iou = 0
                    best_gt_value = None
                    
                    for gt_box_tuple, gt_value in gt_dict[image_id].items():
                        gt_box = list(gt_box_tuple)
                        iou = compute_iou(pred_box, gt_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_value = gt_value
                    
                    if best_iou >= 0.5 and qr['value'] == best_gt_value:
                        total_correct += 1
    
    # Calculate accuracy
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QR code detection and decoding")
    parser.add_argument('--pred_detection', type=str, required=True, help='Path to detection prediction JSON file')
    parser.add_argument('--gt_detection', type=str, required=True, help='Path to detection ground truth JSON file')
    parser.add_argument('--pred_decoding', type=str, help='Path to decoding prediction JSON file (optional)')
    parser.add_argument('--gt_decoding', type=str, help='Path to decoding ground truth JSON file (optional)')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for detection evaluation')
    
    args = parser.parse_args()
    
    # Evaluate detection
    precision, recall, f1_score = evaluate_detection(
        args.pred_detection,
        args.gt_detection,
        args.iou_threshold
    )
    
    print("Detection Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Evaluate decoding if files are provided
    if args.pred_decoding and args.gt_decoding:
        accuracy = evaluate_decoding(args.pred_decoding, args.gt_decoding)
        print("\nDecoding Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")