import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchvision import transforms
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NIH_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass','Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema','Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class InferenceModel:
    def __init__(self, weights_path):
        print(f"--- STARTUP: Loading Model to {DEVICE} ---")
        self.device = DEVICE
        self.model = xrv.models.DenseNet(weights="densenet121-res224-nih")
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 14)
        self.model.op_threshs = None
        self.load_weights(weights_path)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found at {path}")
        state_dict = torch.load(path, map_location=self.device)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(new_state_dict, strict=False)

    def predict(self, img_path):
        try:
            img = xrv.utils.load_image(img_path)
            img = self.transform(img)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(img)
                probs = torch.sigmoid(out).cpu().numpy()[0]

            if len(probs) == len(NIH_CLASSES):
                full_results = dict(zip(NIH_CLASSES, probs))

            else:
                full_results = {f"Class {i}": p for i, p in enumerate(probs)}
            high_confidence = {k: v for k, v in full_results.items() if v > 0.7}

            return full_results, high_confidence

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None 