import numpy as np
from skimage.feature import hog
from skimage import exposure
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os

class TraditionalMLBaseline:
    """
    A class to demonstrate traditional ML approach for X-ray classification.
    Uses HOG (Histogram of Oriented Gradients) features and Random Forest.
    """
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.mlb = MultiLabelBinarizer()
        
    def extract_features(self, image):
        """
        Extract HOG features from an image.
        Args:
            image: Grayscale image (numpy array)
        """
        # Resize to a fixed size for traditional ML
        resized_img = cv2.resize(image, (128, 128))
        
        # Extract HOG features
        fd, _ = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
        return fd

    def preprocess_image(self, image_path):
        """
        Load and grayscale an image.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        return image

    def train(self, X_images, y_labels):
        """
        X_images: List of image paths or numpy arrays
        y_labels: List of lists (pathologies)
        """
        features_list = []
        for img in X_images:
            if isinstance(img, str):
                img_data = self.preprocess_image(img)
            else:
                img_data = img
            features_list.append(self.extract_features(img_data))
            
        X = np.array(features_list)
        y = self.mlb.fit_transform(y_labels)
        
        self.model.fit(X, y)
        print("Model trained successfully.")

    def predict(self, image):
        """
        Predict pathologies for a single image.
        """
        features = self.extract_features(image).reshape(1, -1)
        probs = self.model.predict_proba(features)
        
        # Random Forest multi-output returns a list of arrays for proba
        # We take the probability of the '1' class for each pathology
        predictions = {}
        for i, class_name in enumerate(self.mlb.classes_):
            # probs[i] is [prob_0, prob_1]
            predictions[class_name] = probs[i][0][1] if isinstance(probs, list) else 0.0
            
        return predictions

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model, 'mlb': self.mlb}, path)
        
    def load_model(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.mlb = data['mlb']
