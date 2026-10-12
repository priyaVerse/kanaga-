from flask import Flask, render_template, request
from pathlib import Path
from PIL import Image
import io, time, os
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'results' / 'best_model.pth'
DATASET_DIR = BASE / 'dataset'
UPLOAD_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024

# Put your 23 ImageFolder class directories under dataset/ to use their names.
class_names = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()]) if DATASET_DIR.exists() else []
if not class_names:
    class_names = [f'Class {i}' for i in range(23)]
num_classes = len(class_names)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model():
    if not MODEL_PATH.exists():
        return None, f'Model file not found: {MODEL_PATH}'
    model = build_model()
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        if isinstance(checkpoint, dict):
            checkpoint = {k.removeprefix('module.'): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        model.eval()
        return model, None
    except Exception as e:
        return None, f'Could not load checkpoint. Check that build_model() matches your training model. Details: {e}'

model, model_error = load_model()

def predict(device, tensor):
    model.to(device)
    x = tensor.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    confidence, index = torch.max(probs, dim=1)
    return class_names[int(index.item())] if int(index.item()) < len(class_names) else f'Class {int(index.item())}', float(confidence.item()) * 100, elapsed

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = model_error
    image_data = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            error = 'Choose an image first.'
        elif Path(file.filename).suffix.lower() not in UPLOAD_EXTS:
            error = 'Upload a JPG, PNG, WEBP, or BMP image.'
        elif model is None:
            error = model_error or 'Model is unavailable.'
        else:
            try:
                raw = file.read()
                img = Image.open(io.BytesIO(raw)).convert('RGB')
                import base64
                image_data = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')
                tensor = transform(img).unsqueeze(0)
                cpu_label, cpu_conf, cpu_ms = predict(torch.device('cpu'), tensor)
                if torch.cuda.is_available():
                    gpu_label, gpu_conf, gpu_ms = predict(torch.device('cuda'), tensor)
                    gpu_name = torch.cuda.get_device_name(0)
                else:
                    gpu_label, gpu_conf, gpu_ms = None, None, None
                    gpu_name = 'CUDA GPU not available'
                result = {'label': gpu_label or cpu_label, 'confidence': gpu_conf if gpu_conf is not None else cpu_conf,
                          'cpu_label': cpu_label, 'cpu_confidence': cpu_conf, 'cpu_ms': cpu_ms,
                          'gpu_label': gpu_label, 'gpu_confidence': gpu_conf, 'gpu_ms': gpu_ms,
                          'gpu_name': gpu_name, 'speedup': (cpu_ms / gpu_ms) if gpu_ms and gpu_ms > 0 else None}
            except Exception as e:
                error = f'Could not process image: {e}'
    gpu_available = torch.cuda.is_available()
    return render_template('index.html', result=result, error=error, image_data=image_data,
                           gpu_available=gpu_available, gpu_name=torch.cuda.get_device_name(0) if gpu_available else 'Not available',
                           model_path=str(MODEL_PATH))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
