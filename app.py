from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import traceback
import urllib.request

app = Flask(__name__)
CORS(app, origins="*")


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None

        # Find the last convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.target_layer = module

        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        if self.target_layer is None:
            return None

        try:
            # Forward pass
            self.model.eval()
            output = self.model(input_tensor)

            # Backward pass
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                return None

            # Generate CAM
            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]

            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize to input image size
            cam = cv2.resize(cam, (448, 224))

            return cam
        except Exception as e:
            print(f"GradCAM error: {e}")
            return None


class JawboneAnalyzer:
    def __init__(self, checkpoint_path):
        """Initialize the jawbone analyzer"""
        print("Loading jawbone ensemble...")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract info
        self.architectures = self.checkpoint['architectures_used']
        self.num_classes = self.checkpoint['num_classes']
        self.input_size = self.checkpoint['input_size']
        self.label_mapping = self.checkpoint['label_mapping']
        self.state_dicts = self.checkpoint['model_state_dicts']
        self.cv_scores = self.checkpoint['cv_scores']

        # Create models
        self.models = []
        self._load_models()

        # Calculate ensemble weights
        scores_array = np.array(self.cv_scores)
        self.weights = np.exp(scores_array / 20)
        self.weights = self.weights / self.weights.sum()

        print(f"Loaded ensemble with {len(self.models)} models")
        print(f"CV Scores: {[f'{score:.1f}%' for score in self.cv_scores]}")

    def _load_models(self):
        """Load each model in the ensemble"""
        for i, (arch, state_dict) in enumerate(zip(self.architectures, self.state_dicts)):
            try:
                # Create model using timm
                model = timm.create_model(arch, pretrained=False, num_classes=self.num_classes)

                # Recreate classifier structure
                if hasattr(model, 'fc'):
                    in_features = model.fc.in_features
                    model.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(in_features, self.num_classes)
                    )
                elif hasattr(model, 'classifier'):
                    if hasattr(model.classifier, 'in_features'):
                        in_features = model.classifier.in_features
                        model.classifier = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(in_features, self.num_classes)
                        )
                    else:
                        in_features = model.classifier[-1].in_features
                        model.classifier[-1] = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(in_features, self.num_classes)
                        )

                # Load weights
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                self.models.append(model)
                print(f"Model {i + 1} ({arch}) loaded")
            except Exception as e:
                print(f"Failed to load model {i + 1}: {e}")
                raise

        # Keep only the 3 best models for speed optimization
        if len(self.models) > 3:
            print(f"Reducing from {len(self.models)} to 3 best models for speed...")

            # Get indices of 3 highest CV scores
            best_indices = np.argsort(self.cv_scores)[-3:]

            # Keep only the best models and their data
            self.models = [self.models[i] for i in best_indices]
            self.architectures = [self.architectures[i] for i in best_indices]
            self.cv_scores = [self.cv_scores[i] for i in best_indices]

            print(f"Using top 3 models with CV scores: {[f'{score:.1f}%' for score in self.cv_scores]}")

    def preprocess_image(self, image_data):
        """Preprocess image from base64 data"""
        try:
            # Decode base64
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode image")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to training size
            img_resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))

            # Normalize
            if img_resized.max() > 1.0:
                img_resized = img_resized / 255.0

            # Convert to tensor
            img_tensor = torch.FloatTensor(img_resized).permute(2, 0, 1)

            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_normalized = (img_tensor - mean) / std

            return img_normalized.unsqueeze(0), img_resized

        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def generate_heatmap(self, input_tensor, predicted_class, original_image):
        """Generate attention heatmap"""
        try:
            # Use the model with highest CV score
            best_model_idx = np.argmax(self.cv_scores)
            best_model = self.models[best_model_idx]

            # Generate Grad-CAM
            grad_cam = GradCAM(best_model)
            heatmap = grad_cam.generate_cam(input_tensor, predicted_class)

            if heatmap is None:
                # Fallback: create a simple center-focused heatmap
                h, w = original_image.shape[:2]
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h // 2, w // 2
                heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (min(h, w) / 3) ** 2)
                heatmap = heatmap / heatmap.max()

            # Create overlay image
            heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
            overlay = (original_image * 0.6 + heatmap_colored * 0.4 * 255).astype(np.uint8)

            # Convert to base64
            overlay_pil = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_pil.save(buffer, format='PNG')
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

            return heatmap_base64

        except Exception as e:
            print(f"Heatmap generation error: {e}")
            return None

    def analyze_image(self, image_data):
        """Main analysis function"""
        try:
            # Preprocess image
            input_tensor, original_image = self.preprocess_image(image_data)

            # Get ensemble predictions (no TTA for speed)
            ensemble_output = torch.zeros(1, self.num_classes)

            with torch.no_grad():
                for model, weight in zip(self.models, self.weights):
                    # Single prediction (no TTA)
                    output = model(input_tensor)
                    ensemble_output += weight * F.softmax(output, dim=1)

            # Get final prediction
            probabilities = ensemble_output[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

            # Convert class index to age
            rating_mapping = {v: k for k, v in self.label_mapping.items()}
            predicted_age = rating_mapping[predicted_class]

            # Generate heatmap
            heatmap_base64 = self.generate_heatmap(input_tensor, predicted_class, original_image)

            return {
                'success': True,
                'age': float(predicted_age),
                'confidence': float(confidence),
                'heatmap_base64': heatmap_base64,
                'all_probabilities': probabilities.tolist()
            }

        except Exception as e:
            print(f"Analysis error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# Global analyzer instance
analyzer = None


def download_model():
    """Download model from Dropbox if not already present"""
    MODEL_URL = "https://www.dropbox.com/scl/fi/ziq8fbcx7l8jlk3ea5ofd/jawbone_ensemble.pth?rlkey=y7e51qh7xdvfj5k05x6ml4xzw&st=ndzw14qe&dl=1"
    local_path = "/app/jawbone_ensemble.pth"
    expected_min_size = 100000000  # 100MB minimum

    # Check if model already exists and is valid size
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        print(f"Model already exists: {local_path} ({file_size} bytes)")

        if file_size > expected_min_size:
            return local_path
        else:
            print(f"File too small ({file_size} bytes), re-downloading...")
            os.remove(local_path)

    try:
        print("Downloading model from Dropbox...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        urllib.request.urlretrieve(MODEL_URL, local_path)

        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"Model downloaded successfully! Size: {file_size} bytes")
            return local_path
        else:
            print("Download failed")
            return None

    except Exception as e:
        print(f"Failed to download model: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer is not None
    })


@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """Main analysis endpoint"""
    try:
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded - check server logs for model download status'
            }), 500

        # Get image data from request
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400

        # Analyze image
        result = analyzer.analyze_image(data['image'])

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        print(f"Endpoint error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    model_status = "Loaded" if analyzer is not None else "Not loaded"

    return jsonify({
        'message': 'Jawbone Analysis API',
        'status': 'running',
        'model_status': model_status,
        'endpoints': {
            'health': '/health',
            'analyze': '/analyze (POST)'
        }
    })


def init_model():
    """Initialize the model"""
    global analyzer

    try:
        # First try to download/find the model
        model_path = download_model()

        if not model_path:
            print("Could not download or find model file")
            return False

        # Try to load the model
        analyzer = JawboneAnalyzer(model_path)
        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"Failed to initialize model: {e}")
        traceback.print_exc()
        return False


# Initialize model for gunicorn
print("Starting Jawbone Analysis API...")
init_model()
print("API ready!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)