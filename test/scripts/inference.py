import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchvision import transforms

# --- CONFIGURATION ---
IMG_PATH = "test\img\\00001247_014.png" 
WEIGHTS_PATH = "model/best_nih_densenet121.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Revert to the standard 14 NIH classes (since your weights match this)
NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def predict_image():
    # 1. LOAD IMAGE
    print(f"Loading image from {IMG_PATH}...")
    try:
        img = xrv.utils.load_image(IMG_PATH) 
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    # 2. INITIALIZE MODEL
    print("Initializing base model...")
    # Load base architecture
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    
    # 3. CRITICAL FIX: Handle Class Sizing
    print("Resizing classifier to 14 classes...")
    
    # A. Resize the linear layer to 14 to match your weights
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    
    # B. FIX: Delete op_threshs
    # The original "14 vs 18" error happened because this attribute
    # retained the original 18 values from the library init. 
    # Setting it to None prevents the conflict.
    model.op_threshs = None 
    
    # 4. LOAD WEIGHTS
    print(f"Loading weights from {WEIGHTS_PATH}...")
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return
    
    # Unpack if nested
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    # 5. LOAD STATE DICT
    # We use strict=False because your weights file is missing 'op_threshs'
    # (which is fine, we just deleted it anyway).
    print("Loading state dictionary...")
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(DEVICE)
    model.eval()
    
    # 6. INFERENCE
    print("Running inference...")
    with torch.no_grad():
        out = model(img)
        probs = torch.sigmoid(out).cpu().numpy()[0]

    # 7. RESULTS
    print("\n--- Predictions ---")
    
    if len(probs) == len(NIH_CLASSES):
        results = zip(NIH_CLASSES, probs)
    else:
        results = [(f"Class {i}", p) for i, p in enumerate(probs)]
        
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    for label, prob in sorted_results:
        marker = ">>" if prob > 0.5 else "  "
        print(f"{marker} {label:20s}: {prob:.4f} ({prob*100:.1f}%)")

if __name__ == "__main__":
    predict_image()