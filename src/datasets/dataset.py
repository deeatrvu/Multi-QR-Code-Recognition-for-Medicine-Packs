import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MedicinePackDataset(Dataset):
    """Dataset for medicine pack QR code detection"""
    
    def __init__(self, data_dir, annotation_file=None, transform=None, is_train=True):
        """
        Args:
            data_dir (str): Directory with all the images
            annotation_file (str): Path to annotation file
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): Whether this is training set or not
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Load annotations if provided
        self.annotations = {}
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                for item in annotations:
                    self.annotations[item['image_id']] = item['qrs']
        
        # Get all image files
        self.image_files = []
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_id = os.path.splitext(filename)[0]
                self.image_files.append((image_id, filename))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_id, img_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, img_name)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations if available
        target = {
            'image_id': image_id,
            'boxes': [],
            'labels': []
        }
        
        if image_id in self.annotations:
            for qr in self.annotations[image_id]:
                # Convert [x_min, y_min, x_max, y_max] to tensor
                box = torch.tensor(qr['bbox'], dtype=torch.float32)
                target['boxes'].append(box)
                # All boxes are QR codes (class 1)
                target['labels'].append(torch.tensor(1, dtype=torch.int64))
        
        if len(target['boxes']) > 0:
            target['boxes'] = torch.stack(target['boxes'])
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        
        if len(target['labels']) > 0:
            target['labels'] = torch.stack(target['labels'])
        else:
            target['labels'] = torch.zeros(0, dtype=torch.int64)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor by default
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        return image, target

def get_data_loaders(data_dir, annotation_file, batch_size=4, train_split=0.8, transform=None):
    """
    Create train and validation data loaders
    
    Args:
        data_dir (str): Directory with all the images
        annotation_file (str): Path to annotation file
        batch_size (int): Batch size for data loaders
        train_split (float): Proportion of data to use for training
        transform (callable, optional): Optional transform to be applied on a sample
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create dataset
    dataset = MedicinePackDataset(data_dir, annotation_file, transform, is_train=True)
    
    # Split into train and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    
    # Random split
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """
    Custom collate function for object detection batches
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    return images, targets