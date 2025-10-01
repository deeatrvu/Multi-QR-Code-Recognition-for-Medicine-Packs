import os
import json
import random

def generate_annotations(image_dir, output_file):
    """Generate random annotations for all images in the directory"""
    annotations = []
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_id = os.path.splitext(filename)[0]
            
            # Generate 1-3 random QR codes per image
            num_qrs = random.randint(1, 3)
            qrs = []
            
            for _ in range(num_qrs):
                # Generate random bounding box
                x_min = random.randint(50, 300)
                y_min = random.randint(50, 300)
                width = random.randint(100, 200)
                height = random.randint(100, 200)
                
                qrs.append({
                    "bbox": [x_min, y_min, x_min + width, y_min + height]
                })
            
            annotations.append({
                "image_id": image_id,
                "qrs": qrs
            })
    
    # Save annotations to file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Generated annotations for {len(annotations)} images")

if __name__ == "__main__":
    image_dir = "data/train_images"
    output_file = "data/train_annotations.json"
    generate_annotations(image_dir, output_file)