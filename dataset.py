import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import cv2
import numpy as np
from glob import glob
import random
import pickle
from sklearn.model_selection import train_test_split
import timm

# -------------------------
# 🖼️ Dataset Class
# -------------------------
class DermoscopyDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, labels=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        if self.augment:
            img = self._augment(img)

        if self.mask_paths is not None and self.labels is not None:
            mask = cv2.imread(self.mask_paths[idx], 0)
            mask = cv2.resize(mask, (224, 224))
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, mask, label
        
        return img

    def _augment(self, img):
        """Apply random augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        img = TF.rotate(img, angle)
        
        # Random color jitter
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        img = color_jitter(img)
        
        return img

# -------------------------
# 🧠 Model Architecture with Swin Transformer
# -------------------------
class SwinFeatureExtractor(nn.Module):
    """Swin Transformer feature extractor"""
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3]  # Get features from all stages
        )
        
    def forward(self, x):
        features = self.swin(x)
        # timm Swin models return features in [B, H, W, C] format
        # Convert to [B, C, H, W] format for standard CNN operations
        converted_features = []
        for feat in features:
            if len(feat.shape) == 4 and feat.shape[1] != feat.shape[2]:
                # Assume it's [B, H, W, C], convert to [B, C, H, W]
                feat = feat.permute(0, 3, 1, 2)
            converted_features.append(feat)
        return converted_features

class DermoscopyModel(nn.Module):
    def __init__(self, num_classes=2, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        
        # Swin Transformer backbone
        self.backbone = SwinFeatureExtractor(model_name, pretrained=True)
        
        # Get feature dimensions by running a test forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.backbone(test_input)
            # Convert the last feature to proper format
            last_feature = test_features[-1]
            if len(last_feature.shape) == 4 and last_feature.shape[1] == last_feature.shape[2]:
                last_feature = last_feature.permute(0, 3, 1, 2)
            self.feature_dim = last_feature.shape[1]  # Channel dimension
            self.feature_spatial_size = last_feature.shape[2]  # Spatial dimension
        
        # Segmentation decoder - progressive upsampling
        # Start from the spatial size we get from Swin (e.g., 7x7)
        current_channels = self.feature_dim
        
        self.seg_decoder = nn.Sequential(
            # First upsample: 7x7 -> 14x14
            nn.ConvTranspose2d(current_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Second upsample: 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Third upsample: 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Fourth upsample: 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Fifth upsample: 112x112 -> 224x224
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final 1x1 conv to get segmentation mask
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Classification head - use global average pooling on final features
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract multi-scale features from Swin Transformer
        features = self.backbone(x)
        
        # Use the highest resolution feature map for segmentation
        seg_features = features[-1]  # Shape: [B, H, W, C] from Swin
        
        # Convert from [B, H, W, C] to [B, C, H, W]
        if len(seg_features.shape) == 4 and seg_features.shape[1] == seg_features.shape[2]:
            # This is [B, H, W, C], convert to [B, C, H, W]
            seg_features = seg_features.permute(0, 3, 1, 2)
        
        # Apply segmentation decoder
        seg_out = self.seg_decoder(seg_features)
        
        # For classification, we need to handle the feature format too
        cls_features = features[-1]
        if len(cls_features.shape) == 4 and cls_features.shape[1] == cls_features.shape[2]:
            cls_features = cls_features.permute(0, 3, 1, 2)
        
        # Global classification using the converted features
        cls_out = self.class_head(cls_features)
        
        return seg_out, cls_out

# -------------------------
# 📊 Training Utilities
# -------------------------
class Trainer:
    def __init__(self, model, device=None):
        if device is None:
            # Use MPS if available (Apple Silicon), otherwise CPU
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        
        # Loss functions
        self.seg_criterion = nn.BCEWithLogitsLoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        
        # Store best model state
        self.best_model_state = None
        
    def train(self, train_loader, val_loader, epochs=10, lr=1e-4):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_seg_loss = 0.0
            train_cls_loss = 0.0
            
            # Create progress bar if tqdm is available
            try:
                from tqdm import tqdm
                data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            except ImportError:
                data_iter = train_loader
                print(f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (images, masks, labels) in enumerate(data_iter):
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                seg_pred, cls_pred = self.model(images)
                
                loss_seg = self.seg_criterion(seg_pred, masks)
                loss_cls = self.cls_criterion(cls_pred, labels)
                loss = loss_seg + loss_cls
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_seg_loss += loss_seg.item()
                train_cls_loss += loss_cls.item()
                
                # Print progress every 10 batches if no tqdm
                if 'tqdm' not in str(type(data_iter)) and batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            val_loss, val_seg_loss, val_cls_loss = self.validate(val_loader)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ New best model saved! Validation loss: {val_loss:.4f}")
            
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss/len(train_loader):.4f} '
                  f'(Seg: {train_seg_loss/len(train_loader):.4f}, '
                  f'Cls: {train_cls_loss/len(train_loader):.4f}) | '
                  f'Val Loss: {val_loss:.4f} '
                  f'(Seg: {val_seg_loss:.4f}, Cls: {val_cls_loss:.4f}) | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        return self.best_model_state
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                seg_pred, cls_pred = self.model(images)
                loss_seg = self.seg_criterion(seg_pred, masks)
                loss_cls = self.cls_criterion(cls_pred, labels)
                total_loss = loss_seg + loss_cls
                
                val_loss += total_loss.item()
                val_seg_loss += loss_seg.item()
                val_cls_loss += loss_cls.item()
        
        return (val_loss / len(val_loader), 
                val_seg_loss / len(val_loader), 
                val_cls_loss / len(val_loader))

# -------------------------
# 🚀 Main Training Function
# -------------------------
def train_and_save_model(data_dir, save_path='dermoscopy_swin_model.pkl', epochs=10, batch_size=8):
    print("🏥 Starting Dermoscopy Model Training with Swin Transformer")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Save path: {save_path}")
    
    # Prepare data paths - support multiple image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(data_dir, ext)))
    image_paths = sorted(image_paths)
    
    print(f"📊 Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    # Check if mask files exist
    mask_paths = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(os.path.dirname(data_dir), "generated_masks", f"{base_name}.png")
        mask_paths.append(mask_path)
    
    # Check if mask files exist, create dummy masks if not
    masks_exist = all(os.path.exists(mask_path) for mask_path in mask_paths)
    if not masks_exist:
        print("⚠️  Warning: Mask files not found. Creating dummy masks.")
        mask_dir = os.path.join(os.path.dirname(data_dir), "generated_masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        # Create dummy masks (black circles in the center)
        for i, img_path in enumerate(image_paths):
            mask_path = mask_paths[i]
            
            # Only create if it doesn't exist
            if not os.path.exists(mask_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    center = (w // 2, h // 2)
                    radius = min(h, w) // 4
                    cv2.circle(mask, center, radius, 255, -1)
                    cv2.imwrite(mask_path, mask)
                    if i < 5:  # Print first few
                        print(f"  Created mask: {os.path.basename(mask_path)}")
    
    # Create dummy labels (binary classification)
    # You can modify this to use real labels if available
    labels = [random.randint(0, 1) for _ in image_paths]
    print(f"🏷️  Generated {len(labels)} labels (modify for real labels)")
    
    # Split data
    train_img, val_img, train_mask, val_mask, train_lbl, val_lbl = train_test_split(
        image_paths, mask_paths, labels, test_size=0.2, random_state=42)
    
    print(f"📚 Train samples: {len(train_img)}, Validation samples: {len(val_img)}")
    
    # Create datasets
    train_dataset = DermoscopyDataset(train_img, train_mask, train_lbl, augment=True)
    val_dataset = DermoscopyDataset(val_img, val_mask, val_lbl)
    
    # Create dataloaders with smaller batch size for memory efficiency
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # No multiprocessing to reduce memory
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model and trainer
    print("🏗️  Initializing Swin Transformer model...")
    model = DermoscopyModel(num_classes=2, model_name='swin_tiny_patch4_window7_224')
    trainer = Trainer(model)
    
    # Train model
    print("🚀 Starting training...")
    best_model_state = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model state")
    
    # Save model as pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'💾 Model saved to {save_path}')
    
    return model

# -------------------------
# 🧪 Test Function
# -------------------------
def test_model_loading(model_path):
    """Test loading and using the saved model"""
    print(f"🧪 Testing model loading from {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    model.eval()
    
    # Test with dummy input - make sure it's on the same device as model
    device = next(model.parameters()).device
    test_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        seg_out, cls_out = model(test_input)
        print(f"✅ Model test passed!")
        print(f"   Segmentation output shape: {seg_out.shape}")
        print(f"   Classification output shape: {cls_out.shape}")
    
    return model

if __name__ == '__main__':
    # Main training
    try:
        model = train_and_save_model(
            data_dir='data/input_data',
            save_path='dermoscopy_swin_model.pkl',
            epochs=5,  # Reduced for testing
            batch_size=4  # Small batch size for memory efficiency
        )
        
        # Test the saved model
        test_model_loading('dermoscopy_swin_model.pkl')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
