#!/usr/bin/env python3
"""
Model Conversion Script: MPS to CPU Compatible
==============================================

This script loads the MPS-saved model and re-saves it in a CPU-compatible format
that can be loaded in Docker containers.
"""

import torch
import pickle
import sys
import os

# Import the model classes
try:
    from dataset import DermoscopyModel, SwinFeatureExtractor
    print("✅ Model classes imported successfully")
except ImportError as e:
    print(f"❌ Error importing model classes: {e}")
    sys.exit(1)

def convert_model_to_cpu_compatible(input_path, output_path):
    """
    Convert MPS-saved model to CPU-compatible format
    """
    try:
        print(f"🔄 Loading model from {input_path}...")
        
        # Try to load on MPS first (if available)
        if torch.backends.mps.is_available():
            print("📱 MPS available - loading on MPS device")
            with open(input_path, 'rb') as f:
                model = pickle.load(f)
            
            # Move model to CPU
            print("💻 Converting model to CPU...")
            model = model.cpu()
            
        else:
            print("💻 MPS not available - attempting CPU load with map_location")
            # If MPS not available, try various loading strategies
            try:
                # Method 1: Direct torch.load with map_location
                model = torch.load(input_path, map_location='cpu', weights_only=False)
            except Exception as e1:
                print(f"❌ Method 1 failed: {e1}")
                try:
                    # Method 2: Load with pickle and manual device mapping
                    import io
                    with open(input_path, 'rb') as f:
                        buffer = f.read()
                    buffer_io = io.BytesIO(buffer)
                    model = torch.load(buffer_io, map_location='cpu', weights_only=False)
                except Exception as e2:
                    print(f"❌ Method 2 failed: {e2}")
                    raise Exception(f"All loading methods failed. Original errors: {e1}, {e2}")
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Save in CPU-compatible format
        print(f"💾 Saving CPU-compatible model to {output_path}...")
        
        # Save using torch.save instead of pickle for better compatibility
        torch.save(model, output_path, pickle_protocol=2)
        
        print("✅ Model conversion completed successfully!")
        
        # Verify the converted model can be loaded
        print("🔍 Verifying converted model...")
        test_model = torch.load(output_path, map_location='cpu', weights_only=False)
        print("✅ Verification successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    input_model = "dermoscopy_swin_model.pkl"
    output_model = "dermoscopy_swin_model_cpu.pkl"
    
    print("🔧 Model Conversion Utility")
    print("=" * 40)
    
    if not os.path.exists(input_model):
        print(f"❌ Input model file {input_model} not found!")
        return False
    
    success = convert_model_to_cpu_compatible(input_model, output_model)
    
    if success:
        print("\n🎉 Conversion completed!")
        print(f"📁 Original model: {input_model}")
        print(f"📁 CPU-compatible model: {output_model}")
        print("\n💡 Next steps:")
        print("1. Use the new CPU-compatible model in Docker")
        print("2. Update your app to load from the new file")
        print("3. Test the Docker container")
    else:
        print("\n❌ Conversion failed!")
        print("The model might need to be retrained or saved differently.")
    
    return success

if __name__ == "__main__":
    main()