# Setup Guide

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
