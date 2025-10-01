import os
import torch
from src.models.detector import QRCodeDetector

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Create a dummy model
model = QRCodeDetector(num_classes=2, pretrained=False)

# Save the model
model.save('outputs/dummy_model.pth')
print("Dummy model created and saved to outputs/dummy_model.pth")