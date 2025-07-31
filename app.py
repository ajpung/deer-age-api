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

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.target_layer = module

        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        if self.target_layer is None:
            return None

        try:
            self.model.eval()
            output = self.model(input_tensor)

            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                return None

            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]

            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            input_height, input_width = input_tensor.shape[2], input_tensor.shape[3]
            cam = cv2.resize(cam, (input_width, input_height))

            return cam
        except Exception as e:
            print(f"GradCAM error: {e}")
            return None


class DeerAnalyzer:
    def __init__(self, checkpoint_path, model_name):
        print(f"Loading {model_name} ensemble...")
        self.model_name = model_name
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Debug: print available keys
        print(f"Available keys in {model_name} checkpoint: {list(self.checkpoint.keys())}")

        # Handle different checkpoint formats
        if 'architectures_used' in self.checkpoint:
            # New format (jawbone)
            self.architectures = self.checkpoint['architectures_used']
            self.num_classes = self.checkpoint['num_classes']
            self.input_size = self.checkpoint['input_size']
            self.label_mapping = self.checkpoint['label_mapping']
            self.state_dicts = self.checkpoint['model_state_dicts']
            self.cv_scores = self.checkpoint['cv_scores']
        else:
            # Old format or different structure - try common alternatives
            self.architectures = self.checkpoint.get('architectures',
                                                     self.checkpoint.get('model_architectures', ['resnet50'] * 5))
            self.num_classes = self.checkpoint.get('num_classes', self.checkpoint.get('n_classes', 10))
            self.input_size = self.checkpoint.get('input_size', [224, 224])
            self.label_mapping = self.checkpoint.get('label_mapping', self.checkpoint.get('class_mapping', {}))
            self.state_dicts = self.checkpoint.get('model_state_dicts', self.checkpoint.get('state_dicts',
                                                                                            self.checkpoint.get(
                                                                                                'models', [])))
            self.cv_scores = self.checkpoint.get('cv_scores', self.checkpoint.get('scores', [95.0] * 5))

            # Handle single model case
            if 'state_dict' in self.checkpoint and not self.state_dicts:
                self.state_dicts = [self.checkpoint['state_dict']]
                self.architectures = ['resnet50']
                self.cv_scores = [95.0]

            # If still missing critical info, use trail camera defaults
            if not self.label_mapping:
                # Create default mapping for trail camera (assuming classes 0-9 for ages 1-10)
                self.label_mapping = {str(i + 1): i for i in range(self.num_classes)}

            if not isinstance(self.state_dicts, list):
                self.state_dicts = [self.state_dicts]

            if not isinstance(self.architectures, list):
                self.architectures = [self.architectures]

            if not isinstance(self.cv_scores, list):
                self.cv_scores = [self.cv_scores]

            # Ensure we have 5 ResNet-50 models for trail camera
            if self.model_name == 'trailcam' and len(self.architectures) == 1:
                self.architectures = ['resnet50'] * len(self.state_dicts)

            print(f"Using fallback structure: {len(self.state_dicts)} models, architectures: {self.architectures}")

        self.models = []
        self._load_models()

        scores_array = np.array(self.cv_scores)
        self.weights = np.exp(scores_array / 20)
        self.weights = self.weights / self.weights.sum()

        print(f"Loaded {model_name} ensemble with {len(self.models)} models")
        print(f"CV Scores: {[f'{score:.1f}%' for score in self.cv_scores]}")

    def _load_models(self):
        for i, (arch, state_dict) in enumerate(zip(self.architectures, self.state_dicts)):
            try:
                model = timm.create_model(arch, pretrained=False, num_classes=self.num_classes)

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

                model.load_state_dict(state_dict, strict=True)
                model.eval()
                self.models.append(model)
                print(f"Model {i + 1} ({arch}) loaded")
            except Exception as e:
                print(f"Failed to load model {i + 1}: {e}")
                raise

        if len(self.models) > 1:
            print(f"Reducing from {len(self.models)} to 1 best model for maximum speed...")

            best_index = np.argmax(self.cv_scores)

            self.models = [self.models[best_index]]
            self.architectures = [self.architectures[best_index]]
            self.cv_scores = [self.cv_scores[best_index]]

            print(f"Using single best model with CV score: {self.cv_scores[0]:.1f}%")

    def preprocess_image(self, image_data):
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode image")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))

            if img_resized.max() > 1.0:
                img_resized = img_resized / 255.0

            img_tensor = torch.FloatTensor(img_resized).permute(2, 0, 1)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_normalized = (img_tensor - mean) / std

            return img_normalized.unsqueeze(0), img_resized

        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def generate_heatmap(self, input_tensor, predicted_class, original_image):
        try:
            best_model_idx = np.argmax(self.cv_scores)
            best_model = self.models[best_model_idx]

            grad_cam = GradCAM(best_model)
            heatmap = grad_cam.generate_cam(input_tensor, predicted_class)

            if heatmap is None:
                h, w = original_image.shape[:2]
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h // 2, w // 2
                heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (min(h, w) / 3) ** 2)
                heatmap = heatmap / heatmap.max()

            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            overlay = (original_image * 0.6 + heatmap_colored * 0.4 * 255).astype(np.uint8)

            overlay_pil = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_pil.save(buffer, format='PNG')
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

            return heatmap_base64

        except Exception as e:
            print(f"Heatmap generation error: {e}")
            return None

    def analyze_image(self, image_data, include_heatmap=False):
        try:
            input_tensor, original_image = self.preprocess_image(image_data)

            ensemble_output = torch.zeros(1, self.num_classes)

            with torch.no_grad():
                for model, weight in zip(self.models, self.weights):
                    output = model(input_tensor)
                    ensemble_output += weight * F.softmax(output, dim=1)

            probabilities = ensemble_output[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

            rating_mapping = {v: k for k, v in self.label_mapping.items()}
            predicted_age = rating_mapping[predicted_class]

            heatmap_base64 = None
            if include_heatmap:
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


jawbone_analyzer = None
trailcam_analyzer = None


def download_model(model_type):
    if model_type == 'jawbone':
        MODEL_URL = "https://www.dropbox.com/scl/fi/ziq8fbcx7l8jlk3ea5ofd/jawbone_ensemble.pth?rlkey=y7e51qh7xdvfj5k05x6ml4xzw&st=ndzw14qe&dl=1"
        local_path = "/app/jawbone_ensemble.pth"
    else:  # trailcam
        MODEL_URL = "https://www.dropbox.com/scl/fi/mlxzmdxmbsva2xcjmk0aq/trailcam_ensemble.pth?rlkey=j20g65643vogy0etiyrlnbz97&dl=1"
        local_path = "/app/trailcam_ensemble.pth"

    expected_min_size = 100000000

    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        print(f"{model_type} model already exists: {local_path} ({file_size} bytes)")

        if file_size > expected_min_size:
            return local_path
        else:
            print(f"File too small ({file_size} bytes), re-downloading...")
            os.remove(local_path)

    try:
        print(f"Downloading {model_type} model from Dropbox...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        urllib.request.urlretrieve(MODEL_URL, local_path)

        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"{model_type} model downloaded successfully! Size: {file_size} bytes")
            return local_path
        else:
            print(f"{model_type} download failed")
            return None

    except Exception as e:
        print(f"Failed to download {model_type} model: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'jawbone_loaded': jawbone_analyzer is not None,
        'trailcam_loaded': trailcam_analyzer is not None
    })


@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400

        model_type = data.get('model_type', 'jawbone')
        include_heatmap = data.get('include_heatmap', False)

        if model_type == 'jawbone':
            if jawbone_analyzer is None:
                return jsonify({
                    'success': False,
                    'error': 'Jawbone model not loaded'
                }), 500
            analyzer = jawbone_analyzer
        elif model_type == 'trailcam':
            if trailcam_analyzer is None:
                return jsonify({
                    'success': False,
                    'error': 'Trail camera model not loaded'
                }), 500
            analyzer = trailcam_analyzer
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown model type: {model_type}'
            }), 400

        result = analyzer.analyze_image(data['image'], include_heatmap)

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
    jawbone_status = "Loaded" if jawbone_analyzer is not None else "Not loaded"
    trailcam_status = "Loaded" if trailcam_analyzer is not None else "Not loaded"

    return jsonify({
        'message': 'Deer Age Analysis API',
        'status': 'running',
        'models': {
            'jawbone': jawbone_status,
            'trailcam': trailcam_status
        },
        'endpoints': {
            'health': '/health',
            'analyze': '/analyze (POST)'
        }
    })


def init_models():
    global jawbone_analyzer, trailcam_analyzer

    try:
        print("Initializing models...")

        # Load jawbone model
        jawbone_path = download_model('jawbone')
        if jawbone_path:
            jawbone_analyzer = DeerAnalyzer(jawbone_path, 'jawbone')
            print("Jawbone model loaded successfully!")
        else:
            print("Failed to load jawbone model")

        # Load trailcam model
        trailcam_path = download_model('trailcam')
        if trailcam_path:
            trailcam_analyzer = DeerAnalyzer(trailcam_path, 'trailcam')
            print("Trail camera model loaded successfully!")
        else:
            print("Failed to load trail camera model")

        if jawbone_analyzer is None and trailcam_analyzer is None:
            print("Failed to load any models")
            return False

        print("Model initialization complete!")
        return True

    except Exception as e:
        print(f"Failed to initialize models: {e}")
        traceback.print_exc()
        return False


print("Starting Deer Age Analysis API...")
init_models()
print("API ready!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)