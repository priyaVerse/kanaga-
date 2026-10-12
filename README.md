# CUDA-Accelerated Underwater Object Classification

Flask dashboard for uploading an image, showing the predicted class/confidence, and comparing per-upload CPU and CUDA GPU inference times.

## Project files
- `app.py` — Flask server, model loading, CPU/GPU inference timing
- `templates/index.html` — dashboard
- `static/style.css` — Deep Ocean navy/cyan styling
- `static/app.js` — image preview
- `results/best_model.pth` — copy your trained model checkpoint here
- `dataset/` — optionally copy your ImageFolder class folders here so the app can read class names

## Setup on Windows
1. Extract this ZIP into your project folder, for example `D:\AI_Projects\underwater_cuda_project`.
2. Copy your existing trained checkpoint to `results\best_model.pth` (create `results` if needed).
3. Install a matching PyTorch + torchvision CUDA build using the official selector at https://pytorch.org/get-started/locally/ . Verify with `py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"`.
4. Install Flask and Pillow: `py -m pip install Flask Pillow`.
5. Run from the project folder: `py app.py`.
6. Open `http://127.0.0.1:5000` in your browser.

## Important model note
This frontend assumes the trained model is ResNet-18 with a final layer sized to the number of classes. It supports checkpoints saved as a state dictionary or dictionaries containing `state_dict` / `model_state_dict`. If your training code used a different architecture or class ordering, update `build_model()` and `class_names` in `app.py` to match it exactly. For correct labels, keep the same sorted ImageFolder class names/order used during training; copy those class folders under `dataset/` if you want the app to infer names automatically.

The displayed timings measure the model forward pass only, separately on CPU and GPU, and synchronize CUDA around the timed region. Image decoding and preprocessing are not included.
