import cv2
import numpy as np
from pyzbar.pyzbar import decode

def decode_qr_codes(image, boxes):
    """
    Decode QR codes from image using detected bounding boxes
    
    Args:
        image: The image as a numpy array (H, W, C)
        boxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
        
    Returns:
        List of decoded values (or None if decoding failed)
    """
    results = []
    
    for box in boxes:
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        
        # Extract the region of interest
        roi = image[y_min:y_max, x_min:x_max]
        
        # Try to decode
        decoded_objects = decode(roi)
        
        if decoded_objects:
            # Get the first decoded object
            decoded_data = decoded_objects[0].data.decode('utf-8')
            results.append(decoded_data)
        else:
            # If decoding failed, try with some preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Try different thresholds
            for threshold in [100, 150, 200]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                decoded_objects = decode(binary)
                
                if decoded_objects:
                    decoded_data = decoded_objects[0].data.decode('utf-8')
                    results.append(decoded_data)
                    break
            else:
                # If all attempts failed
                results.append(None)
    
    return results

def classify_qr_code(value):
    """
    Classify QR code based on its content
    
    Args:
        value: Decoded QR code value
        
    Returns:
        Classification type (e.g., 'manufacturer', 'batch', 'distributor', 'regulator')
    """
    if value is None:
        return None
    
    # Simple rule-based classification
    # This is a placeholder - in a real implementation, you would use more sophisticated rules
    # or a trained classifier based on the actual data patterns
    
    value = value.upper()
    
    if value.startswith('MFR') or value.startswith('MANF'):
        return 'manufacturer'
    elif value.startswith('B') and any(c.isdigit() for c in value):
        return 'batch'
    elif value.startswith('DIST') or value.startswith('D'):
        return 'distributor'
    elif value.startswith('REG') or value.startswith('R'):
        return 'regulator'
    else:
        return 'unknown'