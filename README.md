# How to Run NVIDIA LocateAnything-3B Locally (Fixing Series 50 GPU Errors)

There are known to be a lot of errors and compatibility issues when trying to run the `nvidia/LocateAnything-3B` model locally, especially on newer NVIDIA Series 50 GPUs. If you've been struggling with hallucinated outputs, cache errors, or broken generation loops, this repository is the solution.

With this guide, anyone can successfully set up and run the LocateAnything-3B model locally without Gradio. By simply pinning the correct dependencies and using our refactored script, you can generate flawless bounding box detections.

## Setup Guide

1. **Create a Clean Environment**
```bash
conda create -n locateanything python=3.12 -y
conda activate locateanything
```

2. **Install PyTorch**
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

3. **Install Critical Dependencies (Exact Transformers version required)**
```bash
pip install transformers==4.57.1
```

4. **Install Remaining Libraries**
```bash
pip install opencv-python-headless Pillow numpy peft huggingface_hub decord lmdb
```

5. **Run Inference**
Update `IMAGE_TO_PROCESS` and `TARGET_CATEGORIES` in `main.py`, then run:
```bash
python main.py
```
