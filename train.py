import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from src.models.detector import QRCodeDetector
from src.datasets.dataset import get_data_loaders
from src.utils.utils import save_checkpoint

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        args.annotation_file,
        batch_size=args.batch_size
    )
    
    # Create model
    model = QRCodeDetector(num_classes=2, pretrained=True)
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Train for one epoch
        model.train()
        epoch_loss = 0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        progress_bar = tqdm(train_loader)
        
        for images, targets in progress_bar:
            # Move to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += losses.item()
            progress_bar.set_description(f"Loss: {losses.item():.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, is_best=True, filename=os.path.join(args.output_dir, 'checkpoint.pth'))
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'final_model.pth'))
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QR code detector")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--lr_step_size', type=int, default=3, help='LR scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR scheduler gamma')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)