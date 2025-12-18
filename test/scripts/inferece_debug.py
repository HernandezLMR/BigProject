import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_PATH = 'best_nih_densenet121.pth' # Use your FIXED model
TEST_IMAGE = "test\\00001247_014.png"         # Your test image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LOAD MODEL (Your code) ---
def get_model(path):
    model = models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(in_channels=1, out_channels=64, 
                                    kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    state = torch.load(path, map_location='cpu', weights_only=True)
    # Handle state dict keys
    if 'state_dict' in state: state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

# --- EXPERIMENTS ---
def run_diagnostic():
    model = get_model(MODEL_PATH)

    weight_std = model.classifier.weight.std().item()
    weight_mean = model.classifier.weight.mean().item()

    print(f"Classifier Weight STD: {weight_std:.6f}")
    print(f"Classifier Weight Mean: {weight_mean:.6f}")
    
    # Base transform WITHOUT normalization (we add it manually to test variants)
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load raw
    raw_img = Image.open(TEST_IMAGE).convert('RGB')
    tensor_base = base_transform(raw_img)
    
    experiments = {
        "1. Normal (0-1, Normalized)": 
            transforms.Normalize([0.5], [0.5])(tensor_base).unsqueeze(0),
            
        "2. Inverted Colors (Negatives)": 
            transforms.Normalize([0.5], [0.5])(1.0 - tensor_base).unsqueeze(0),
            
        "3. Scaled 0-255 (Not Normalized)": 
            (tensor_base * 255.0).unsqueeze(0),
            
        "4. Un-Normalized (Raw 0-1)": 
            tensor_base.unsqueeze(0)
    }

    print(f"\n--- DIAGNOSTIC REPORT FOR {TEST_IMAGE} ---")
    
    for name, input_batch in experiments.items():
        input_batch = input_batch.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_batch)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        top_idx = probs.argmax()
        top_prob = probs[top_idx]
        print(f"\nExperiment: {name}")
        print(f"  > Input Range: [{input_batch.min():.2f}, {input_batch.max():.2f}]")
        print(f"  > Top Prediction: Index {top_idx} ({top_prob:.4f})")
        print(f"  > Variance of all probs: {probs.var():.6f}") 
        # Low variance = model returns flat 'average' for everything
        
        # Save what the model 'sees' to disk for YOU to check
        if name == "1. Normal (0-1, Normalized)":
            debug_img = input_batch.squeeze().cpu().numpy()
            # Un-normalize for viewing: x * 0.5 + 0.5
            debug_img = (debug_img * 0.5) + 0.5 
            plt.imsave("debug_view_normal.png", debug_img, cmap='gray')

    print("\nCheck 'debug_view_normal.png'. If it looks pure black or white, PIL is failing to load your image correctly.")

if __name__ == "__main__":
    run_diagnostic()