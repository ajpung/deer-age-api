from flask import Flask, request, jsonify, send_file
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
from datetime import datetime
import glob

app = Flask(__name__)
CORS(app, origins="*")


class GradCAM:
    def __init__(self, model, model_type="jawbone"):
        self.model = model
        self.model_type = model_type
        self.target_layer = None
        self.gradients = None
        self.activations = None

        # Different target layer selection based on model type
        if model_type == "trailcam":
            # ResNet-50 specific target layer for trailcam
            try:
                self.target_layer = model.layer4[-1].conv3
                print(f"TrailCam: Using ResNet-50 specific target layer: layer4[-1].conv3")
            except Exception as e:
                print(f"TrailCam: Failed to use ResNet-50 layer, falling back to automatic: {e}")
                self._use_automatic_selection()
        else:
            # Keep existing automatic selection for jawbone (working)
            self._use_automatic_selection()

        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_full_backward_hook(self.save_gradient)

    def _use_automatic_selection(self):
        """Original automatic target layer selection"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.target_layer = module
        print(f"{self.model_type}: Using automatic target layer selection")

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

        # Handle the ACTUAL checkpoint format for trailcam
        if 'model_architecture' in self.checkpoint:  # Your actual trailcam format
            # Use the actual saved architecture for all models
            single_arch = self.checkpoint['model_architecture']  # 'resnet50'
            num_models = self.checkpoint.get('num_models', 5)
            self.architectures = [single_arch] * num_models  # ['resnet50'] * 5
            self.num_classes = self.checkpoint['num_classes']
            self.input_size = (224, 224)  # Standard for ResNet-50
            self.label_mapping = self.checkpoint['label_mapping']
            self.state_dicts = self.checkpoint['model_state_dicts']
            self.cv_scores = self.checkpoint['cv_scores']
            print(f"DEBUG: Using actual checkpoint format for {model_name}")
        elif 'architectures_used' in self.checkpoint:  # Legacy jawbone format
            self.architectures = self.checkpoint['architectures_used']
            self.num_classes = self.checkpoint['num_classes']
            self.input_size = self.checkpoint['input_size']
            self.label_mapping = self.checkpoint['label_mapping']
            self.state_dicts = self.checkpoint['model_state_dicts']
            self.cv_scores = self.checkpoint['cv_scores']
        else:
            # Fallback for older formats
            self.architectures = self.checkpoint.get('architectures', ['resnet50'] * 5)
            self.num_classes = self.checkpoint.get('num_classes', 10)
            self.input_size = self.checkpoint.get('input_size', [224, 224])
            self.label_mapping = self.checkpoint.get('label_mapping', {})
            self.state_dicts = self.checkpoint.get('model_state_dicts', [])
            self.cv_scores = self.checkpoint.get('cv_scores', [95.0] * 5)

            if 'state_dict' in self.checkpoint and not self.state_dicts:
                self.state_dicts = [self.checkpoint['state_dict']]
                self.architectures = ['resnet50']
                self.cv_scores = [95.0]

            if not self.label_mapping:
                self.label_mapping = {"1.5": 0, "2.5": 1, "3.5": 2, "4.5": 3, "5.5": 4}

            if not isinstance(self.state_dicts, list):
                self.state_dicts = [self.state_dicts]
            if not isinstance(self.architectures, list):
                self.architectures = [self.architectures]
            if not isinstance(self.cv_scores, list):
                self.cv_scores = [self.cv_scores]

        print(f"DEBUG: {model_name} architectures: {self.architectures}")
        print(f"DEBUG: {model_name} using input_size: {self.input_size}")

        # Convert list to tuple to match working jawbone format
        if isinstance(self.input_size, list):
            self.input_size = tuple(self.input_size)
            print(f"DEBUG: Converted {model_name} input_size to tuple: {self.input_size}")

        self.models = []
        self._load_models()

        scores_array = np.array(self.cv_scores)
        self.weights = np.exp(scores_array / 20)
        self.weights = self.weights / self.weights.sum()

        print(f"Loaded {model_name} ensemble with {len(self.models)} models")

    def _load_models(self):
        for i, (arch, state_dict) in enumerate(zip(self.architectures, self.state_dicts)):
            model = timm.create_model(arch, pretrained=False, num_classes=self.num_classes)

            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, self.num_classes))
            elif hasattr(model, 'classifier'):
                if hasattr(model.classifier, 'in_features'):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, self.num_classes))

            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.models.append(model)

    def preprocess_image(self, image_data):
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # DEBUG: Save raw received image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filename = f"/tmp/{self.model_name}_raw_received_{timestamp}.png"
        Image.fromarray(img).save(raw_filename)
        print(f"DEBUG: Saved raw received image: {raw_filename}")
        print(f"DEBUG: Raw image shape: {img.shape}")

        img_resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        original_image = img_resized.copy()

        if img_resized.max() > 1.0:
            img_resized = img_resized / 255.0

        img_tensor = torch.FloatTensor(img_resized).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_normalized = (img_tensor - mean) / std

        return img_normalized.unsqueeze(0), original_image

    def generate_heatmap(self, input_tensor, predicted_class, original_image):
        try:
            best_model_idx = np.argmax(self.cv_scores)
            best_model = self.models[best_model_idx]

            # Pass model_type to GradCAM
            grad_cam = GradCAM(best_model, self.model_name)
            heatmap = grad_cam.generate_cam(input_tensor, predicted_class)

            if heatmap is None:
                print(f"ERROR: GradCAM failed for {self.model_name}")
                return None

            # Save individual components to /tmp/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Original processed image
            original_filename = f"/tmp/{self.model_name}_original_{timestamp}.png"
            Image.fromarray(original_image.astype(np.uint8)).save(original_filename)

            # Raw heatmap
            heatmap_colored = cm.jet(heatmap)
            heatmap_img = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            heatmap_filename = f"/tmp/{self.model_name}_heatmap_{timestamp}.png"
            Image.fromarray(heatmap_img).save(heatmap_filename)

            # Final overlay
            overlay = self.create_processed_heatmap_overlay(original_image, heatmap)
            overlay_filename = f"/tmp/{self.model_name}_overlay_{timestamp}.png"
            Image.fromarray(overlay).save(overlay_filename)

            print(f"DEBUG: Saved {self.model_name} images:")
            print(f"  Original: {original_filename}")
            print(f"  Heatmap: {heatmap_filename}")
            print(f"  Overlay: {overlay_filename}")

            overlay_pil = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_pil.save(buffer, format='PNG')
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

            return heatmap_base64

        except Exception as e:
            print(f"Heatmap generation error: {e}")
            return None

    def create_processed_heatmap_overlay(self, original_image, heatmap):
        heatmap_thresh = np.copy(heatmap)
        heatmap_thresh[heatmap_thresh < 0.001] = 0

        if heatmap_thresh.max() > 0:
            heatmap_thresh = heatmap_thresh / heatmap_thresh.max()

        heatmap_colored = cm.jet(heatmap_thresh)
        heatmap_colored_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

        alpha_channel = heatmap_thresh.copy()
        alpha_channel = (alpha_channel * 255).astype(np.uint8)

        adjusted_original = self.apply_brightness_contrast(original_image, 0, -50)

        overlay = adjusted_original.copy().astype(np.float32)
        mask = alpha_channel > 0

        if np.any(mask):
            alpha_norm = alpha_channel[mask].astype(np.float32) / 255.0
            base = overlay[mask] / 255.0
            blend = heatmap_colored_rgb[mask].astype(np.float32) / 255.0

            overlay_blend = np.zeros_like(base)
            dark_mask = base <= 0.5
            overlay_blend[dark_mask] = 2 * base[dark_mask] * blend[dark_mask]
            overlay_blend[~dark_mask] = 1 - 2 * (1 - base[~dark_mask]) * (1 - blend[~dark_mask])

            overlay[mask] = (base * (1 - alpha_norm[:, np.newaxis]) +
                             overlay_blend * alpha_norm[:, np.newaxis]) * 255

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def apply_brightness_contrast(self, image, brightness=0, contrast=0):
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

        return np.clip(image, 0, 255).astype(np.uint8)

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
            return {'success': False, 'error': str(e)}


jawbone_analyzer = None
trailcam_analyzer = None


def download_model(model_type):
    if model_type == 'jawbone':
        MODEL_URL = "https://www.dropbox.com/scl/fi/ziq8fbcx7l8jlk3ea5ofd/jawbone_ensemble.pth?rlkey=y7e51qh7xdvfj5k05x6ml4xzw&st=ndzw14qe&dl=1"
        local_path = "/app/jawbone_ensemble.pth"
    else:
        MODEL_URL = "https://www.dropbox.com/scl/fi/mlxzmdxmbsva2xcjmk0aq/trailcam_ensemble.pth?rlkey=j20g65643vogy0etiyrlnbz97&dl=1"
        local_path = "/app/trailcam_ensemble.pth"

    if os.path.exists(local_path) and os.path.getsize(local_path) > 100000000:
        return local_path

    print(f"Downloading {model_type} model...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, local_path)
    return local_path


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'jawbone_loaded': jawbone_analyzer is not None,
        'trailcam_loaded': trailcam_analyzer is not None
    })


@app.route('/debug_files', methods=['GET'])
def list_debug_files():
    files = glob.glob('/tmp/*.png')
    file_list = [os.path.basename(f) for f in files]
    return jsonify({'files': file_list})


@app.route('/debug_file/<filename>', methods=['GET'])
def get_debug_file(filename):
    filepath = f'/tmp/{filename}'
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        model_type = data.get('model_type', 'jawbone')
        include_heatmap = data.get('include_heatmap', False)

        if model_type == 'jawbone':
            analyzer = jawbone_analyzer
        elif model_type == 'trailcam':
            analyzer = trailcam_analyzer
        else:
            return jsonify({'success': False, 'error': f'Unknown model type: {model_type}'}), 400

        if analyzer is None:
            return jsonify({'success': False, 'error': f'{model_type} model not loaded'}), 500

        result = analyzer.analyze_image(data['image'], include_heatmap)
        return jsonify(result)

    except Exception as e:
        print(f"Endpoint error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Deer Age Analysis API',
        'status': 'running',
        'models': {
            'jawbone': "Loaded" if jawbone_analyzer else "Not loaded",
            'trailcam': "Loaded" if trailcam_analyzer else "Not loaded"
        }
    })


def init_models():
    global jawbone_analyzer, trailcam_analyzer

    try:
        print("Initializing models...")

        jawbone_path = download_model('jawbone')
        if jawbone_path:
            jawbone_analyzer = DeerAnalyzer(jawbone_path, 'jawbone')
            print("Jawbone model loaded!")

        trailcam_path = download_model('trailcam')
        if trailcam_path:
            trailcam_analyzer = DeerAnalyzer(trailcam_path, 'trailcam')
            print("Trail camera model loaded!")

        print("Models ready!")
        return True

    except Exception as e:
        print(f"Failed to initialize models: {e}")
        return False


print("Starting API...")
init_models()
print("API ready!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)