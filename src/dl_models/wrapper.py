import torch
import radiology_engine as re
import numpy as np
import cv2

class DLModelWrapper:
    """
    Inference engine for Deep Learning models.
    """
    def __init__(self, weights="densenet121-res224-all"):
        self.model = re.models.DenseNet(weights=weights)
        self.model.eval()
        self.pathologies = self.model.pathologies

    def preprocess(self, image):
        """
        Standardizes image for clinical model analysis.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Standardize normalization
        img = re.datasets.normalize(image, 255)
        
        # Resize and Center Crop
        img = img[None, ...] # Add channel dim
        
        # Simple resize
        img_resized = cv2.resize(img[0], (224, 224))
        img_tensor = torch.from_numpy(img_resized[None, ...]).float()
        
        return img_tensor

    def predict(self, image):
        """
        Runs inference on a single image.
        Args:
            image: numpy array (grayscale or BGR)
        Returns:
            Dictionary of pathology probabilities
        """
        img_tensor = self.preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(img_tensor[None, ...])
            
        probs = outputs[0].detach().numpy()
        return dict(zip(self.pathologies, probs))
