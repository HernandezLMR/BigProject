import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import os

# --- CONFIGURATION ---
BROKEN_MODEL_PATH = 'best_nih_densenet121.pth'
FIXED_MODEL_OUTPUT = 'fixed_densenet121.pth'
IMAGES_FOLDER = 'data\images_001\images'  # Point this to any folder with X-rays
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Define Model (Same as your original) ---
def get_model(path):
    model = models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(in_channels=1, out_channels=64, 
                                    kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 14)
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=True)
    model.to(DEVICE)
    return model

# --- 2. Define Transform (Must match your inference transform) ---
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

# --- 3. Simple Dataset for the fix ---
class FixDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.files = glob.glob(os.path.join(image_dir, "*.*"))
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        # We only need about 50-100 images to fix the stats
        self.files = self.files[:200] 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

# --- 4. The Repair Logic ---
if __name__ == "__main__":
    print(f"Loading broken model: {BROKEN_MODEL_PATH}")
    model = get_model(BROKEN_MODEL_PATH)
    
    print(f"Loading images from: {IMAGES_FOLDER}")
    dataset = FixDataset(IMAGES_FOLDER, get_transform())
    
    if len(dataset) == 0:
        print("Error: No images found in the folder. Cannot fix model.")
        exit()
        
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # CRITICAL STEP: Switch to train mode so BatchNorm updates
    model.train() 
    
    print("Running recalibration (Forward passes only)...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            model(batch) # Forward pass updates running_mean/var
            print(f"  Processed batch {i+1}/{len(loader)}")
            
    print("Recalibration complete.")
    
    # Save the repaired state_dict
    torch.save(model.state_dict(), FIXED_MODEL_OUTPUT)
    print(f"SUCCESS! Fixed model saved to: {FIXED_MODEL_OUTPUT}")