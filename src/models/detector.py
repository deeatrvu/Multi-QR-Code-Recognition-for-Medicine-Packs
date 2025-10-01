import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class QRCodeDetector:
    """QR Code detector model based on Faster R-CNN"""
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the QR code detector
        
        Args:
            num_classes (int): Number of classes (background + QR code)
            pretrained (bool): Whether to use pretrained backbone
        """
        # Load a pre-trained model for the backbone
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        
        # FasterRCNN needs to know the number of output channels in the backbone
        backbone.out_channels = 2048
        
        # Define anchor generator
        # QR codes can have various sizes, so we use multiple sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define the RoI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=800,
            max_size=1333,
            box_score_thresh=0.7,  # Higher threshold for confidence
            box_nms_thresh=0.3,    # Lower threshold for NMS to avoid removing overlapping QR codes
        )
    
    def train(self, mode=True):
        """Set the model to training mode"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set the model to evaluation mode"""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move the model to the specified device"""
        self.model.to(device)
        return self
    
    def __call__(self, images, targets=None):
        """
        Forward pass through the model
        
        Args:
            images (List[Tensor]): Images to be processed
            targets (List[Dict], optional): Ground-truth boxes and labels
            
        Returns:
            loss_dict (Dict) or detections (List[Dict]): 
                During training, returns losses
                During inference, returns detections
        """
        return self.model(images, targets)
    
    def save(self, path):
        """Save the model to disk"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path, device='cpu'):
        """Load the model from disk"""
        self.model.load_state_dict(torch.load(path, map_location=device))
        return self