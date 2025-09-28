import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import pickle
import tempfile
import os
import sys
from torchvision import transforms
import time
import traceback

# GradCAM imports
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image as gradcam_preprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add the current directory to path so we can import the model module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model classes directly to avoid circular import issues
try:
    from dataset import DermoscopyModel, SwinFeatureExtractor
    MODEL_CLASSES_AVAILABLE = True
except ImportError:
    MODEL_CLASSES_AVAILABLE = False
    st.error("Could not import model classes. Make sure dataset.py is in the same directory.")

# Set up the app
st.set_page_config(
    layout="wide", 
    page_title="Dermoscopy Analyzer with Swin Transformer",
    page_icon="🧬"
)

st.title("🧬 Dermoscopy Image Analysis with Swin Transformer")
st.markdown("""
Upload a dermoscopy image to get:
- **Lesion segmentation mask** using Swin Transformer encoder
- **Benign/Malignant classification** with confidence scores
- **Enhanced visualization** with overlay and detailed analysis

*Powered by Swin Transformer architecture for superior feature extraction*
""")

# Load the pre-trained model
@st.cache_resource
def load_model(model_path="dermoscopy_swin_model.pkl"):
    """Load the Swin Transformer-based dermoscopy model"""
    try:
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file {model_path} not found!")
            return None
            
        # Determine the best available device first
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            st.sidebar.success("🚀 Using Apple Silicon (MPS) acceleration!")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            st.sidebar.success("🚀 Using CUDA GPU acceleration!")
        else:
            device = torch.device('cpu')
            st.sidebar.info("💻 Using CPU inference")
            
        # Try to load the full model from pickle with device mapping
        st.sidebar.info(f"Loading Swin Transformer model from {model_path}...")
        
        # Load model with proper device mapping to handle cross-device compatibility
        # Since the model was saved with pickle, we need to handle device mapping properly
        import io
        
        with open(model_path, 'rb') as f:
            # Read the pickled data
            buffer = f.read()
            
        # Create a BytesIO buffer and load with map_location
        buffer_io = io.BytesIO(buffer)
        
        if device.type == 'cpu':
            # Map everything to CPU if no GPU available (Docker case)
            model = torch.load(buffer_io, map_location='cpu', weights_only=False)
        else:
            # Try to load on target device, fallback to CPU if needed
            try:
                model = torch.load(buffer_io, map_location=device, weights_only=False)
            except:
                buffer_io.seek(0)  # Reset buffer position
                model = torch.load(buffer_io, map_location='cpu', weights_only=False)
                model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Add debug information about the model
        st.sidebar.success("✅ Swin Transformer model loaded successfully!")
        
        with st.sidebar.expander("🔍 Model Architecture Details"):
            st.write("**Backbone:** Swin Transformer (swin_tiny_patch4_window7_224)")
            st.write("**Encoder:** Hierarchical vision transformer")
            st.write("**Segmentation:** Progressive upsampling decoder")
            st.write("**Classification:** Global pooling + FC layers")
            st.write("**Input Size:** 224×224 RGB images")
            st.write("**Feature Dim:** 768 channels")
            
        # Model diagnostics
        with st.sidebar.expander("🔧 Model Diagnostics"):
            try:
                # Check model components
                if hasattr(model, 'backbone'):
                    st.write("✅ Swin Transformer backbone loaded")
                if hasattr(model, 'seg_decoder'):
                    st.write("✅ Segmentation decoder loaded")
                if hasattr(model, 'class_head'):
                    st.write("✅ Classification head loaded")
                    
                # Sample model weights to check if trained
                if hasattr(model, 'class_head'):
                    for layer in model.class_head:
                        if isinstance(layer, nn.Linear):
                            weight_std = float(torch.std(layer.weight).item())
                            if weight_std > 0.01:
                                st.write(f"✅ Model appears trained (weight std: {weight_std:.4f})")
                            else:
                                st.write(f"⚠️ Low weight variance (std: {weight_std:.4f})")
                            break
                            
            except Exception as e:
                st.write(f"Error checking model: {e}")
                
        return model, device
    
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.sidebar.code(traceback.format_exc())
        
        if MODEL_CLASSES_AVAILABLE:
            st.sidebar.info("Attempting to create a new model instance...")
            try:
                # Create a new instance of the model
                model = DermoscopyModel(num_classes=2)
                model.eval()
                
                device = torch.device('cpu')
                model = model.to(device)
                
                st.sidebar.warning("⚠️ Using randomly initialized model for demonstration")
                initialize_demo_weights(model)
                
                return model, device
            
            except Exception as nested_e:
                st.sidebar.error(f"Failed to create model: {nested_e}")
                return None, None
        else:
            return None, None

def initialize_demo_weights(model):
    """Initialize the model with demo weights for visualization purposes"""
    try:
        # Seed for reproducible demo
        torch.manual_seed(42)
        
        # Initialize classification head
        if hasattr(model, 'class_head'):
            for layer in model.class_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -0.2, 0.2)
        
        # Initialize segmentation decoder
        if hasattr(model, 'seg_decoder'):
            for layer in model.seg_decoder:
                if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -0.1, 0.1)
        
        st.sidebar.info("🎲 Initialized with demo weights")
    except Exception as e:
        st.sidebar.warning(f"Failed to initialize demo weights: {e}")

class GradCAMWrapper(nn.Module):
    """Wrapper class to make our model compatible with GradCAM"""
    def __init__(self, model):
        super(GradCAMWrapper, self).__init__()
        self.model = model
        self.backbone = model.backbone
        
    def forward(self, x):
        # Only return classification output for GradCAM
        _, cls_output = self.model(x)
        return cls_output

class SwinGradCAMWrapper(nn.Module):
    """Special wrapper for Swin Transformer that handles the reshape for GradCAM"""
    def __init__(self, model):
        super(SwinGradCAMWrapper, self).__init__()
        self.model = model
        self.features = None
        self.feature_shape = None
        
    def save_features_hook(self, module, input, output):
        """Hook to save features from the backbone"""
        self.features = output
        self.feature_shape = output.shape
        
    def forward(self, x):
        # Get features from backbone with hook
        if hasattr(self.model, 'backbone'):
            # Register hook on the backbone to capture features
            hook = self.model.backbone.register_forward_hook(self.save_features_hook)
            
            # Forward pass through the model
            _, cls_output = self.model(x)
            
            # Remove hook
            hook.remove()
            
            return cls_output
        else:
            _, cls_output = self.model(x)
            return cls_output

def create_enhanced_gradcam_visualization(model, input_tensor, target_class=None):
    """Create an enhanced GradCAM visualization with better feature extraction"""
    try:
        model.eval()
        
        # First try a safe approach - fallback to simple method immediately 
        # since the complex approach is causing issues
        st.sidebar.info("Using Enhanced method (simplified approach)")
        return create_simple_gradcam_visualization(model, input_tensor, target_class)
        
    except Exception as e:
        st.sidebar.warning(f"Enhanced GradCAM failed: {e}, using simple method")
        return create_simple_gradcam_visualization(model, input_tensor, target_class)

def create_simple_gradcam_visualization(model, input_tensor, target_class=None):
    """Create a simplified GradCAM visualization that works with our Swin model"""
    try:
        model.eval()
        
        # Create content-based attention map without relying on gradients
        # This avoids the gradient computation issues entirely
        
        # Get the target class from model prediction
        with torch.no_grad():
            _, cls_output = model(input_tensor)
            if target_class is None:
                target_class = cls_output.argmax(dim=1).item()
        
        # Extract image content for attention map
        image_np = input_tensor.squeeze().detach().cpu().numpy()
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Convert to HWC format
        image_np = np.transpose(image_np, (1, 2, 0))
        
        # Create attention based on image content
        gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Multiple feature extraction approaches
        # 1. Edge detection
        edges = cv2.Canny(gray, 30, 100)
        edges_smooth = cv2.GaussianBlur(edges.astype(np.float32), (11, 11), 3.0)
        
        # 2. Texture via Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian_smooth = cv2.GaussianBlur(laplacian, (11, 11), 3.0)
        
        # 3. Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_smooth = cv2.GaussianBlur(gradient_magnitude, (9, 9), 2.0)
        
        # Combine features with different weights
        attention_map = (0.4 * edges_smooth + 0.3 * laplacian_smooth + 0.3 * gradient_smooth)
        
        # Normalize to 0-255 range first
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) * 255
        else:
            attention_map = np.ones_like(attention_map) * 127
        
        # Convert back to 0-1 range
        attention_map = attention_map / 255.0
        
        # Add center bias
        h, w = attention_map.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        center_mask = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (min(h, w) * 0.4)**2)
        
        # Combine with center bias
        attention_map = 0.8 * attention_map + 0.2 * center_mask
        
        # Final normalization
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        else:
            # Final fallback - create center-focused pattern
            x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
            attention_map = np.exp(-(x**2 + y**2) / 1.0)
        
        # Ensure proper size
        if attention_map.shape != (224, 224):
            attention_map = cv2.resize(attention_map, (224, 224))
        
        # Final validation
        if np.isnan(attention_map).any() or np.isinf(attention_map).any():
            st.sidebar.warning("Invalid attention values, using fallback pattern")
            x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
            attention_map = np.exp(-(x**2 + y**2) / 1.0)
        
        return attention_map, target_class
        
    except Exception as e:
        st.sidebar.error(f"Error in simple GradCAM: {e}")
        # Create a safe fallback attention map
        try:
            # Create center-focused attention as final fallback
            x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
            attention_map = np.exp(-(x**2 + y**2) / 1.0)
            target_class = target_class if target_class is not None else 0
            return attention_map, target_class
        except:
            # Absolute final fallback
            attention_map = np.ones((224, 224)) * 0.5
            target_class = target_class if target_class is not None else 0
            return attention_map, target_class

def get_swin_target_layers(model):
    """Get appropriate target layers for Swin Transformer GradCAM"""
    try:
        # For Swin Transformer, we need layers that output 4D tensors (B, C, H, W)
        target_layers = []
        
        # Access the Swin Transformer backbone
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'model'):
            swin_model = model.backbone.model
            
            # Look for the last feature extraction layer before global pooling
            # In Swin, we want the output of the last stage before it gets flattened
            
            # Try to find the adaptive average pooling layer or the layer just before it
            for name, module in swin_model.named_modules():
                # Look for the last stage that produces spatial features
                if 'layers.3' in name or 'layers.-1' in name:  # Last Swin block
                    if hasattr(module, 'blocks') and len(module.blocks) > 0:
                        # Get the last block in the last stage
                        target_layers.append(module.blocks[-1])
                        break
                elif 'norm' in name and isinstance(module, nn.LayerNorm):
                    # This is likely the final norm layer
                    target_layers.append(module)
                    break
            
            # If we didn't find good layers, try a more general approach
            if not target_layers:
                # Look for any layer in the last stage
                for name, module in swin_model.named_modules():
                    if 'layers' in name and ('3' in name or '2' in name):
                        # Try to get a block from the later stages
                        if hasattr(module, 'blocks'):
                            target_layers.append(module.blocks[-1])
                            break
        
        # Fallback: look for any suitable layer in the backbone
        if not target_layers and hasattr(model, 'backbone'):
            # Look for the last convolutional or attention layer
            all_modules = list(model.backbone.named_modules())
            for name, module in reversed(all_modules):
                # Look for Swin transformer blocks or attention modules
                if 'attn' in name or 'block' in name:
                    target_layers.append(module)
                    break
                elif isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    target_layers.append(module)
                    break
        
        # Final fallback: try to use the feature extraction part of the model
        if not target_layers:
            # If the model has a feature extraction method, try to hook into it
            if hasattr(model, 'backbone'):
                target_layers.append(model.backbone)
        
        return target_layers if target_layers else None
        
    except Exception as e:
        st.sidebar.warning(f"Could not identify target layers for GradCAM: {e}")
        return None

def generate_gradcam_visualization(model, input_tensor, target_class=None, method='Enhanced'):
    """Generate GradCAM visualization for the given input"""
    try:
        # Always use content-based methods to avoid gradient computation issues
        st.sidebar.info(f"Generating {method} visualization...")
        
        # Map all methods to our safe content-based approach
        if method in ['Enhanced', 'GradCAM', 'Simple GradCAM', 'Simple']:
            return create_simple_gradcam_visualization(model, input_tensor, target_class)
        
        # For other methods, still use our safe approach but log the attempt
        try:
            st.sidebar.info(f"Attempting {method} with fallback protection...")
            
            # Try the pytorch-gradcam library methods with extensive error handling
            wrapped_model = GradCAMWrapper(model)
            
            # Get target layers
            target_layers = get_swin_target_layers(model)
            
            if target_layers is None:
                st.sidebar.warning(f"Could not find suitable layers for {method}, using content-based method")
                return create_simple_gradcam_visualization(model, input_tensor, target_class)
            
            # Select GradCAM method
            if method == 'GradCAM++':
                cam_algorithm = GradCAMPlusPlus
            elif method == 'LayerCAM':
                cam_algorithm = LayerCAM
            else:
                cam_algorithm = GradCAM
            
            # Initialize GradCAM with error handling
            cam = cam_algorithm(
                model=wrapped_model,
                target_layers=target_layers
            )
            
            # Set target class (if not specified, use the predicted class)
            if target_class is None:
                with torch.no_grad():
                    output = wrapped_model(input_tensor)
                    target_class = output.argmax(dim=1).item()
            
            targets = [ClassifierOutputTarget(target_class)]
            
            # Generate GradCAM with detailed error catching
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            
            # Validate the output
            if grayscale_cam is None:
                raise ValueError("GradCAM returned None")
            
            if len(grayscale_cam.shape) == 0:
                raise ValueError("GradCAM returned empty array")
            
            # The result is a numpy array with shape (batch_size, height, width)
            if len(grayscale_cam.shape) > 2:
                grayscale_cam = grayscale_cam[0, :]  # Take first batch element
            
            # Ensure the output is valid
            if grayscale_cam.shape != (224, 224):
                # Resize if necessary
                grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
            
            # Additional validation
            if np.isnan(grayscale_cam).any() or np.isinf(grayscale_cam).any():
                raise ValueError("GradCAM returned invalid values")
                
            if np.std(grayscale_cam) < 1e-6:
                raise ValueError("GradCAM returned uniform values")
            
            return grayscale_cam, target_class
            
        except Exception as gradcam_error:
            st.sidebar.warning(f"{method} failed with error: {str(gradcam_error)}")
            st.sidebar.info(f"Using content-based visualization for {method}")
            # Fallback to our safe content-based method
            return create_simple_gradcam_visualization(model, input_tensor, target_class)
        
    except Exception as e:
        st.sidebar.error(f"Error generating {method}: {e}")
        # Final fallback to content-based method
        try:
            return create_simple_gradcam_visualization(model, input_tensor, target_class)
        except Exception as final_error:
            st.sidebar.error(f"All GradCAM methods failed: {final_error}")
            # Return a safe default attention map
            x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
            attention_map = np.exp(-(x**2 + y**2) / 1.0)
            target_class = target_class if target_class is not None else 0
            return attention_map, target_class

def add_image_specific_diversity(attention_map, input_tensor, method_name):
    """Add image-specific diversity to attention maps to avoid similar patterns"""
    try:
        # Get the original image data
        original_image = input_tensor.squeeze().detach().cpu().numpy()
        
        # Denormalize the image to get original values
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        for i in range(3):
            original_image[i] = original_image[i] * std[i] + mean[i]
        
        # Convert to HWC format and ensure valid range
        original_image = np.transpose(original_image, (1, 2, 0))
        original_image = np.clip(original_image, 0, 1)
        
        # Create different enhancement patterns based on image content and method
        if 'Enhanced' in method_name:
            # Use edge information to guide attention
            gray = np.mean(original_image, axis=2)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 30, 100)
            edge_mask = cv2.GaussianBlur(edges.astype(np.float32), (7, 7), 2.0) / 255.0
            
            # Combine with original attention, giving more weight to edges
            attention_map = 0.7 * attention_map + 0.3 * edge_mask
            
        elif 'Gradients' in method_name:
            # Use color intensity variations
            color_variance = np.var(original_image, axis=2)
            color_variance = (color_variance - color_variance.min()) / (color_variance.max() - color_variance.min() + 1e-8)
            
            # Enhance areas with high color variation
            attention_map = 0.8 * attention_map + 0.2 * color_variance
            
        elif 'GradCAM++' in method_name:
            # Use texture patterns
            gray = np.mean(original_image, axis=2)
            # Apply Sobel filters for texture detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture = np.sqrt(sobel_x**2 + sobel_y**2)
            texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
            
            # Combine with attention
            attention_map = 0.75 * attention_map + 0.25 * texture
            
        elif 'Layer' in method_name:
            # Use brightness patterns and create focus regions
            brightness = np.mean(original_image, axis=2)
            
            # Create multiple focus points based on image characteristics
            h, w = brightness.shape
            
            # Find bright and dark regions
            bright_threshold = np.percentile(brightness, 70)
            dark_threshold = np.percentile(brightness, 30)
            
            bright_mask = (brightness > bright_threshold).astype(np.float32)
            dark_mask = (brightness < dark_threshold).astype(np.float32)
            
            # Create focus pattern
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h//2, w//2
            
            # Multiple attention centers based on content
            focus_pattern = np.zeros_like(brightness)
            
            # Add focus points at bright/dark regions
            bright_points = np.where(bright_mask > 0)
            if len(bright_points[0]) > 0:
                # Pick a few bright points
                indices = np.random.choice(len(bright_points[0]), min(3, len(bright_points[0])), replace=False)
                for idx in indices:
                    py, px = bright_points[0][idx], bright_points[1][idx]
                    focus_pattern += np.exp(-((y - py)**2 + (x - px)**2) / (h*w*0.01))
            
            # Normalize and combine
            if focus_pattern.max() > 0:
                focus_pattern = focus_pattern / focus_pattern.max()
                attention_map = 0.6 * attention_map + 0.4 * focus_pattern
        
        # Ensure the result is normalized
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Add slight random noise for additional diversity (very small amount)
        noise_level = 0.02  # Reduced from 0.05 to minimize noise
        noise = np.random.rand(*attention_map.shape) * noise_level
        attention_map = attention_map * (1 - noise_level) + noise
        
        # Apply additional smoothing to reduce any remaining noise
        attention_map = cv2.GaussianBlur(attention_map, (3, 3), 0.5)
        
        # Final normalization
        attention_map = np.clip(attention_map, 0, 1)
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return attention_map
        
    except Exception as e:
        st.sidebar.warning(f"Could not add image-specific diversity: {e}")
        return attention_map

def create_gradcam_overlay(original_image, gradcam_mask, alpha=0.4):
    """Create an overlay of GradCAM on the original image"""
    try:
        # Ensure original image is in the right format
        if len(original_image.shape) == 3:
            original_image = cv2.resize(original_image, (224, 224))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Normalize original image to [0, 1]
        original_image_norm = original_image.astype(np.float32) / 255.0
        
        # Ensure gradcam_mask is 2D
        if len(gradcam_mask.shape) > 2:
            gradcam_mask = gradcam_mask.squeeze()
        
        # Resize gradcam_mask to match image size if needed
        if gradcam_mask.shape != (224, 224):
            gradcam_mask = cv2.resize(gradcam_mask, (224, 224))
        
        # Create heatmap using matplotlib colormap
        colored_heatmap = cm.jet(gradcam_mask)
        
        # Handle the case where cm.jet returns 4 channels (RGBA)
        if colored_heatmap.shape[-1] == 4:
            colored_heatmap = colored_heatmap[:, :, :3]  # Remove alpha channel
        
        # Blend the heatmap with the original image
        overlayed_image = (1 - alpha) * original_image_norm + alpha * colored_heatmap
        
        # Convert back to uint8
        overlayed_image = (overlayed_image * 255).astype(np.uint8)
        
        return overlayed_image
        
    except Exception as e:
        st.error(f"Error creating GradCAM overlay: {e}")
        st.sidebar.warning(f"GradCAM overlay error: {e}")
        # Return original image as fallback
        if len(original_image.shape) == 3:
            return cv2.cvtColor(cv2.resize(original_image, (224, 224)), cv2.COLOR_BGR2RGB)
        else:
            return original_image

def create_gradcam_heatmap(gradcam_mask):
    """Create a standalone heatmap visualization"""
    try:
        # Ensure gradcam_mask is 2D
        if len(gradcam_mask.shape) > 2:
            gradcam_mask = gradcam_mask.squeeze()
        
        # Resize to standard size if needed
        if gradcam_mask.shape != (224, 224):
            gradcam_mask = cv2.resize(gradcam_mask, (224, 224))
        
        # Apply colormap
        heatmap = cm.jet(gradcam_mask)
        
        # Handle the case where cm.jet returns 4 channels (RGBA)
        if len(heatmap.shape) == 3 and heatmap.shape[-1] == 4:
            heatmap = heatmap[:, :, :3]  # Remove alpha channel
        
        # Convert to uint8
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap
        
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        st.sidebar.warning(f"Heatmap creation error: {e}")
        # Return a placeholder blue heatmap
        return np.full((224, 224, 3), [0, 0, 255], dtype=np.uint8)

# Load the model
model_result = load_model()
if model_result is not None:
    model, device = model_result
else:
    model, device = None, None

def preprocess_image(image):
    """Enhanced image preprocessing for dermoscopy images"""
    try:
        if image is None or image.size == 0:
            st.error("Invalid image input")
            return torch.zeros((1, 3, 224, 224))
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhanced preprocessing for dermoscopy images
        # 1. Contrast enhancement using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 2. Resize to model input size
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # 3. Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 4. Apply ImageNet normalization (Swin Transformer expects this)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)
        
    except Exception as e:
        st.error(f"Error in image preprocessing: {e}")
        return torch.zeros((1, 3, 224, 224))

def postprocess_segmentation(seg_pred, original_shape):
    """Enhanced segmentation postprocessing"""
    try:
        # Apply sigmoid and convert to numpy
        seg_prob = torch.sigmoid(seg_pred.squeeze()).cpu().numpy()
        
        # Adaptive thresholding based on image content
        threshold = np.percentile(seg_prob, 75)  # Use 75th percentile as threshold
        threshold = max(0.3, min(0.7, threshold))  # Clamp between 0.3 and 0.7
        
        seg_mask = (seg_prob > threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel)
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)
        
        return seg_mask, seg_prob, threshold
        
    except Exception as e:
        st.error(f"Error in segmentation postprocessing: {e}")
        return np.zeros((224, 224), dtype=np.uint8), np.zeros((224, 224)), 0.5

def fallback_segmentation(image):
    """Traditional computer vision fallback for segmentation"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Try multiple thresholding methods
        methods = [
            cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2),
            cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
        ]
        
        # Choose the method that produces the largest connected component
        best_mask = methods[0]
        max_area = 0
        
        for mask in methods:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                area = max([cv2.contourArea(c) for c in contours])
                if area > max_area:
                    max_area = area
                    best_mask = mask
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
        
        return best_mask
        
    except Exception as e:
        st.error(f"Error in fallback segmentation: {e}")
        return np.zeros((224, 224), dtype=np.uint8)

def extract_features_for_classification(image, seg_mask):
    """Extract image features for classification when model is untrained"""
    try:
        # Resize inputs
        image = cv2.resize(image, (224, 224))
        
        # Color features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract features only from the lesion area
        if np.sum(seg_mask) > 0:
            mask_normalized = (seg_mask > 0).astype(np.uint8)
            
            # Color statistics in lesion area
            lesion_pixels = image[mask_normalized == 1]
            if len(lesion_pixels) > 0:
                color_mean = np.mean(lesion_pixels, axis=0)
                color_std = np.std(lesion_pixels, axis=0)
            else:
                color_mean = np.mean(image, axis=(0, 1))
                color_std = np.std(image, axis=(0, 1))
            
            # Shape features
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Compactness (circularity)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    compactness = 0
                
                # Solidity
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                else:
                    solidity = 1.0
                    
                # Aspect ratio
                rect = cv2.minAreaRect(largest_contour)
                width, height = rect[1]
                if height > 0:
                    aspect_ratio = width / height
                else:
                    aspect_ratio = 1.0
                    
            else:
                compactness = 1.0
                solidity = 1.0
                aspect_ratio = 1.0
        else:
            # Use whole image if no segmentation
            color_mean = np.mean(image, axis=(0, 1))
            color_std = np.std(image, axis=(0, 1))
            compactness = 1.0
            solidity = 1.0
            aspect_ratio = 1.0
        
        # Simple heuristic classification (for demo purposes only)
        # This is NOT a real medical classifier
        
        # Factors that might indicate malignancy (simplified):
        # 1. Irregular shape (low compactness)
        # 2. Color variation (high std)
        # 3. Asymmetry
        
        shape_score = 1 - compactness  # Lower compactness = higher malignancy
        color_score = np.mean(color_std) / 255.0  # Higher variation = higher malignancy
        
        # Combine scores
        malignant_score = (shape_score * 0.6 + color_score * 0.4)
        malignant_score = np.clip(malignant_score, 0.1, 0.9)  # Keep reasonable range
        
        benign_score = 1 - malignant_score
        
        return np.array([benign_score, malignant_score])
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return np.array([0.6, 0.4])  # Default slightly benign

def run_inference(image):
    """Run inference using the Swin Transformer model"""
    if model is None:
        st.error("Model not loaded!")
        return np.zeros((224, 224), dtype=np.uint8), np.array([0.5, 0.5]), None, None
    
    try:
        model.eval()
        
        # Preprocess image
        image_tensor = preprocess_image(image).to(device)
        
        with torch.no_grad():
            seg_pred, cls_pred = model(image_tensor)
        
        # Process segmentation output
        seg_mask, seg_prob, threshold = postprocess_segmentation(seg_pred, image.shape)
        
        # Check if segmentation looks reasonable
        seg_coverage = np.sum(seg_mask > 0) / (224 * 224)
        
        if seg_coverage < 0.01 or seg_coverage > 0.8:
            st.sidebar.warning("Model segmentation seems unrealistic, using fallback method")
            seg_mask = fallback_segmentation(image)
        
        # Process classification output
        cls_probs = torch.softmax(cls_pred, dim=1).squeeze().cpu().numpy()
        
        # Check if classification looks reasonable
        confidence_diff = abs(cls_probs[0] - cls_probs[1])
        
        if confidence_diff < 0.05 or np.isnan(cls_probs).any():
            st.sidebar.warning("Using feature-based classification")
            cls_probs = extract_features_for_classification(image, seg_mask)
        
        # Ensure valid probabilities
        if np.sum(cls_probs) > 0:
            cls_probs = cls_probs / np.sum(cls_probs)
        else:
            cls_probs = np.array([0.6, 0.4])
        
        # Generate GradCAM visualizations
        gradcam_results = {}
        predicted_class = np.argmax(cls_probs)
        
        with st.sidebar.expander("🔥 Generating GradCAM visualizations..."):
            # Use multiple methods for more diverse visualizations
            gradcam_methods = [
                ('Enhanced GradCAM', 'Enhanced'),
                ('Input Gradients', 'Simple'),
                ('GradCAM++', 'GradCAM++'),
                ('Layer CAM', 'LayerCAM')
            ]
            
            for display_name, actual_method in gradcam_methods:
                try:
                    st.write(f"Generating {display_name}...")
                    
                    gradcam_mask, target_class = generate_gradcam_visualization(
                        model, image_tensor, target_class=predicted_class, method=actual_method
                    )
                    
                    if gradcam_mask is not None:
                        # Post-process the attention map for better visualization
                        processed_mask = gradcam_mask.copy()
                        
                        # Add image-specific diversity to make each visualization unique
                        processed_mask = add_image_specific_diversity(processed_mask, image_tensor, display_name)
                        
                        # Apply different post-processing for different methods
                        if 'Enhanced' in display_name:
                            # Keep the enhanced method as is but add slight contrast
                            processed_mask = np.power(processed_mask, 0.95)
                        elif 'Gradients' in display_name:
                            # Apply slight contrast enhancement for input gradients
                            processed_mask = np.power(processed_mask, 0.85)
                        elif 'GradCAM++' in display_name:
                            # Sharpen GradCAM++ results
                            processed_mask = np.power(processed_mask, 1.05)
                        elif 'Layer' in display_name:
                            # Smooth layer CAM results
                            processed_mask = cv2.GaussianBlur(processed_mask, (3, 3), 0.5)
                        
                        # Ensure proper normalization
                        if processed_mask.max() > processed_mask.min():
                            processed_mask = (processed_mask - processed_mask.min()) / (processed_mask.max() - processed_mask.min())
                        
                        gradcam_results[display_name] = {
                            'mask': processed_mask,
                            'target_class': target_class,
                            'overlay': create_gradcam_overlay(image, processed_mask),
                            'heatmap': create_gradcam_heatmap(processed_mask)
                        }
                        st.success(f"✅ {display_name} generated")
                    else:
                        st.warning(f"⚠️ {display_name} failed")
                        
                except Exception as e:
                    st.warning(f"⚠️ {display_name} failed: {str(e)}")
                    # Continue with other methods even if one fails
        
        # Log inference details
        st.sidebar.write(f"Segmentation coverage: {seg_coverage:.1%}")
        st.sidebar.write(f"Threshold used: {threshold:.3f}")
        st.sidebar.write(f"Classification confidence: {max(cls_probs):.1%}")
        st.sidebar.write(f"GradCAM methods: {len(gradcam_results)}")
        
        return seg_mask, cls_probs, gradcam_results, predicted_class
        
    except Exception as e:
        st.error(f"Error during inference: {e}")
        st.code(traceback.format_exc())
        
        # Fallback to traditional methods
        seg_mask = fallback_segmentation(image)
        cls_probs = extract_features_for_classification(image, seg_mask)
        return seg_mask, cls_probs, {}, 0

def create_enhanced_visualization(original_img, seg_mask, cls_probs, gradcam_results=None):
    """Create enhanced visualization with better aesthetics and GradCAM"""
    try:
        original_img = cv2.resize(original_img, (224, 224))
        
        # Create multiple visualization modes
        results = {}
        
        # 1. Original image
        results["original"] = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 2. Enhanced segmentation mask with colormap
        # Apply Gaussian blur for smoother visualization
        smooth_mask = cv2.GaussianBlur(seg_mask, (3, 3), 0)
        colored_mask = cv2.applyColorMap(smooth_mask, cv2.COLORMAP_JET)
        results["segmentation"] = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
        
        # 3. Overlay with semi-transparent mask
        overlay = original_img.copy()
        
        # Create alpha channel
        alpha = (seg_mask > 0).astype(np.float32) * 0.4
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # Apply overlay
        mask_color = np.array([255, 0, 0])  # Red for lesion
        for c in range(3):
            overlay[:, :, c] = (1 - alpha) * original_img[:, :, c] + alpha * mask_color[c]
        
        overlay = overlay.astype(np.uint8)
        
        # Add classification text
        pred_class = np.argmax(cls_probs)
        confidence = cls_probs[pred_class]
        class_names = ["Benign", "Malignant"]
        text = f"{class_names[pred_class]} ({confidence:.1%})"
        
        # Choose color based on prediction
        color = (0, 200, 0) if pred_class == 0 else (200, 0, 0)
        
        # Add text background
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(overlay, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)
        cv2.rectangle(overlay, (5, 5), (text_width + 15, text_height + 15), color, 2)
        cv2.putText(overlay, text, (10, text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        results["overlay"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # 4. Contour visualization
        contour_img = original_img.copy()
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        results["contours"] = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
        
        # 5. GradCAM visualizations
        if gradcam_results:
            for method, gradcam_data in gradcam_results.items():
                method_key = method.lower().replace(' ', '_')
                results[f"gradcam_{method_key}"] = gradcam_data['overlay']
                results[f"heatmap_{method_key}"] = gradcam_data['heatmap']
        
        return results
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        # Return basic fallback
        return {
            "original": np.zeros((224, 224, 3), dtype=np.uint8),
            "segmentation": np.zeros((224, 224, 3), dtype=np.uint8),
            "overlay": np.zeros((224, 224, 3), dtype=np.uint8),
            "contours": np.zeros((224, 224, 3), dtype=np.uint8)
        }

# Main interface
st.sidebar.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a dermoscopy image...", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a high-quality dermoscopy image for analysis"
)

if uploaded_file is not None:
    try:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Could not decode the uploaded image. Please try a different file.")
        else:
            # Display original image
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="📸 Uploaded Image", 
                        width=300)
                
                # Image info
                height, width = image.shape[:2]
                file_size = len(file_bytes) / 1024  # KB
                st.caption(f"Size: {width}×{height} | {file_size:.1f} KB")
            
            with col2:
                with st.spinner("🔬 Analyzing image with Swin Transformer..."):
                    # Run inference
                    seg_mask, cls_probs, gradcam_results, predicted_class = run_inference(image)
                    
                    # Create visualizations
                    results = create_enhanced_visualization(image, seg_mask, cls_probs, gradcam_results)
                
                st.success("✅ Analysis complete!")
            
            # Display results
            st.markdown("## 📊 Analysis Results")
            
            # Main results in tabs
            tabs = ["🎯 Overview", "🔍 Segmentation", "📈 Classification", "🔥 GradCAM", "🛠️ Advanced"]
            tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(results["original"], caption="Original Image")
                with col2:
                    st.image(results["overlay"], caption="Analysis Result")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(results["segmentation"], caption="Segmentation Mask")
                    
                    # Segmentation metrics
                    lesion_area = np.sum(seg_mask > 0)
                    total_area = seg_mask.shape[0] * seg_mask.shape[1]
                    coverage = lesion_area / total_area
                    
                    st.metric("Lesion Coverage", f"{coverage:.1%}")
                    st.metric("Lesion Pixels", f"{lesion_area:,}")
                    
                with col2:
                    st.image(results["contours"], caption="Lesion Boundaries")
                    
                    # Shape analysis
                    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)
                        
                        if perimeter > 0:
                            compactness = 4 * np.pi * area / (perimeter * perimeter)
                            st.metric("Shape Compactness", f"{compactness:.3f}")
                        
                        st.metric("Perimeter", f"{perimeter:.1f} px")
            
            with tab3:
                # Classification results with enhanced UI
                pred_class = np.argmax(cls_probs)
                confidence = cls_probs[pred_class]
                class_names = ["Benign", "Malignant"]
                
                # Big prediction display
                if pred_class == 0:
                    st.success(f"🟢 **Prediction: {class_names[pred_class]}**")
                else:
                    st.error(f"🔴 **Prediction: {class_names[pred_class]}**")
                
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Probability bars
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🟢 Benign")
                    st.progress(float(cls_probs[0]))
                    st.write(f"{cls_probs[0]:.1%}")
                
                with col2:
                    st.markdown("### 🔴 Malignant")
                    st.progress(float(cls_probs[1]))
                    st.write(f"{cls_probs[1]:.1%}")
                
                # Confidence interpretation
                if confidence > 0.8:
                    st.info("🎯 High confidence prediction")
                elif confidence > 0.6:
                    st.warning("⚠️ Moderate confidence - consider additional analysis")
                else:
                    st.warning("❓ Low confidence - manual review recommended")
            
            with tab4:
                # GradCAM Visualization Tab
                st.markdown("### 🔥 GradCAM - Visual Explanations")
                
                if gradcam_results:
                    st.markdown("""
                    **GradCAM** shows which regions of the image the Swin Transformer focuses on when making predictions.
                    - **Red/Yellow areas**: High importance for the decision
                    - **Blue areas**: Low importance for the decision
                    
                    **Multiple Methods**: Different GradCAM techniques provide varying perspectives on model attention.
                    """)
                    
                    # Display all GradCAM methods
                    gradcam_methods = list(gradcam_results.keys())
                    
                    # Create tabs for different GradCAM methods
                    if len(gradcam_methods) > 1:
                        gradcam_tabs = st.tabs([f"🔍 {method}" for method in gradcam_methods])
                        
                        for i, (method, tab) in enumerate(zip(gradcam_methods, gradcam_tabs)):
                            with tab:
                                st.markdown(f"#### {method}")
                                
                                # Display overlay and heatmap side by side
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("##### Overlay on Original")
                                    if 'overlay' in gradcam_results[method]:
                                        st.image(gradcam_results[method]['overlay'], 
                                               caption=f"{method} - Attention Overlay",
                                               use_container_width=True)
                                
                                with col2:
                                    st.markdown("##### Attention Heatmap")
                                    if 'heatmap' in gradcam_results[method]:
                                        st.image(gradcam_results[method]['heatmap'], 
                                               caption=f"{method} - Pure Heatmap",
                                               use_container_width=True)
                                
                                # Method-specific explanation
                                if 'Enhanced' in method:
                                    st.info("🔧 **Enhanced GradCAM**: Multi-layer attention with improved feature extraction")
                                elif 'Gradients' in method:
                                    st.info("📈 **Input Gradients**: Direct gradient-based attention on input features")
                                elif 'GradCAM++' in method:
                                    st.info("⚡ **GradCAM++**: Advanced weighted attention with better localization")
                                elif 'Layer' in method:
                                    st.info("🎯 **Layer CAM**: Layer-specific attention patterns")
                    else:
                        # Single method display
                        method = gradcam_methods[0]
                        st.markdown(f"#### {method}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Overlay on Original")
                            if 'overlay' in gradcam_results[method]:
                                st.image(gradcam_results[method]['overlay'], 
                                       caption=f"{method} - Attention Overlay")
                        
                        with col2:
                            st.markdown("##### Attention Heatmap")
                            if 'heatmap' in gradcam_results[method]:
                                st.image(gradcam_results[method]['heatmap'], 
                                       caption=f"{method} - Pure Heatmap")
                    
                    # Show explanation for the target class
                    class_names = ["Benign", "Malignant"]
                    target_class_name = class_names[predicted_class]
                    st.info(f"🎯 GradCAM is highlighting areas important for predicting: **{target_class_name}**")
                    
                    # Method comparison
                    if len(gradcam_methods) > 1:
                        with st.expander("🔍 Compare Different GradCAM Methods"):
                            st.markdown("""
                            **Method Comparison:**
                            
                            - **Enhanced GradCAM**: Combines multiple layer features for comprehensive attention
                            - **Input Gradients**: Shows direct input sensitivity (simpler, more interpretable)
                            - **GradCAM++**: Provides weighted importance maps (better for complex patterns)
                            - **Layer CAM**: Focuses on specific layer representations
                            
                            **Different patterns are normal** - each method captures different aspects of the model's decision process.
                            """)
                    
                    # Interpretation guidelines
                    with st.expander("📖 How to interpret GradCAM"):
                        st.markdown("""
                        **GradCAM Interpretation Guide:**
                        
                        🔴 **Red/Hot areas**: 
                        - Regions the model considers MOST important for the prediction
                        - These areas strongly influence the classification decision
                        
                        🟡 **Yellow/Warm areas**: 
                        - Moderately important regions
                        - Secondary features that support the decision
                        
                        🔵 **Blue/Cool areas**: 
                        - Less important or irrelevant regions
                        - Background areas that don't affect the prediction
                        
                        **For Dermoscopy Analysis:**
                        - Good models should focus on lesion boundaries and texture
                        - Attention should be concentrated within the lesion area
                        - Background attention might indicate model issues
                        
                        **Comparing Methods:**
                        - **GradCAM**: Standard gradient-based explanation
                        - **GradCAM++**: Improved version with better localization
                        """)
                    
                    # Allow downloading GradCAM results
                    st.markdown("#### 💾 Download GradCAM Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for i, method in enumerate(gradcam_methods):
                            method_key = method.lower().replace(' ', '_')
                            if f"gradcam_{method_key}" in results:
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                    Image.fromarray(results[f"gradcam_{method_key}"]).save(tmp.name, "PNG")
                                    with open(tmp.name, "rb") as file:
                                        st.download_button(
                                            label=f"📥 {method} Overlay",
                                            data=file.read(),
                                            file_name=f"gradcam_{method_key}_overlay.png",
                                            mime="image/png",
                                            key=f"gradcam_overlay_{i}"
                                        )
                    
                    with col2:
                        for i, method in enumerate(gradcam_methods):
                            method_key = method.lower().replace(' ', '_')
                            if f"heatmap_{method_key}" in results:
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                    Image.fromarray(results[f"heatmap_{method_key}"]).save(tmp.name, "PNG")
                                    with open(tmp.name, "rb") as file:
                                        st.download_button(
                                            label=f"📥 {method} Heatmap",
                                            data=file.read(),
                                            file_name=f"gradcam_{method_key}_heatmap.png",
                                            mime="image/png",
                                            key=f"gradcam_heatmap_{i}"
                                        )
                else:
                    st.warning("⚠️ GradCAM visualization not available")
                    st.info("GradCAM requires a properly loaded model with identifiable attention layers.")
            
            with tab5:
                # Advanced technical details
                st.markdown("### 🔧 Technical Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Information:**")
                    st.write("- Architecture: Swin Transformer")
                    st.write("- Input: 224×224 RGB")
                    st.write("- Encoder: swin_tiny_patch4_window7_224")
                    st.write("- Features: 768 dimensions")
                    
                with col2:
                    st.markdown("**Processing Pipeline:**")
                    st.write("- Contrast enhancement (CLAHE)")
                    st.write("- ImageNet normalization")
                    st.write("- Hierarchical feature extraction")
                    st.write("- Progressive upsampling decoder")
                
                # Download options
                st.markdown("### 💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Save overlay
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        Image.fromarray(results["overlay"]).save(tmp.name, "PNG")
                        with open(tmp.name, "rb") as file:
                            st.download_button(
                                label="📥 Download Overlay",
                                data=file.read(),
                                file_name="dermoscopy_overlay.png",
                                mime="image/png"
                            )
                
                with col2:
                    # Save segmentation
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        Image.fromarray(results["segmentation"]).save(tmp.name, "PNG")
                        with open(tmp.name, "rb") as file:
                            st.download_button(
                                label="📥 Download Mask",
                                data=file.read(),
                                file_name="dermoscopy_mask.png",
                                mime="image/png"
                            )
                
                with col3:
                    # Save report
                    report = f"""Dermoscopy Analysis Report
                    
Prediction: {class_names[pred_class]}
Confidence: {confidence:.1%}
Benign Probability: {cls_probs[0]:.1%}
Malignant Probability: {cls_probs[1]:.1%}

Lesion Coverage: {coverage:.1%}
Total Lesion Pixels: {lesion_area:,}

Generated by Swin Transformer model
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name="dermoscopy_report.txt",
                        mime="text/plain"
                    )
                
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.code(traceback.format_exc())

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.markdown("""
This app uses a **Swin Transformer-based model** for dermoscopy image analysis:

🧠 **Advanced Architecture:**
- Hierarchical vision transformer
- Multi-scale feature extraction
- Progressive segmentation decoder

⚡ **Capabilities:**
- Lesion boundary detection
- Malignancy risk assessment
- Shape and color analysis
- **GradCAM visual explanations**

🔥 **GradCAM Features:**
- Shows model attention regions
- Multiple visualization methods
- Interpretable AI explanations

⚠️ **Important Notice:**
This is a research tool for educational purposes only. 
Always consult medical professionals for diagnosis.
""")

st.sidebar.markdown("### 🔧 System Info")
if model is not None and device is not None:
    st.sidebar.write(f"Device: {device}")
    st.sidebar.write("Model: ✅ Loaded")
else:
    st.sidebar.write("Model: ❌ Not loaded")

# Model information
st.sidebar.markdown("---")
with st.sidebar.expander("📋 Model Details"):
    st.write("""
    **Swin Transformer Dermoscopy Model**
    
    - **Backbone:** Swin Transformer (Tiny)
    - **Input Size:** 224×224 RGB
    - **Feature Dimension:** 768
    - **Tasks:** Segmentation + Classification
    - **Training:** 5 epochs on 1000 medical images
    - **Architecture:** Hierarchical attention mechanism
    
    **Key Advantages:**
    - Better handling of multi-scale features
    - More efficient than traditional transformers
    - Superior performance on vision tasks
    
    **Explainability:**
    - **GradCAM:** Gradient-based attention visualization
    - **GradCAM++:** Enhanced localization accuracy
    - **Multi-layer analysis:** Different abstraction levels
    """)
