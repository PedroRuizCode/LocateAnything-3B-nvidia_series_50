<div align="center">
  <h1>🎯 How to Run NVIDIA LocateAnything-3B Locally</h1>
  <h3><i>(The Ultimate Fix for NVIDIA Series 50 GPU Errors)</i></h3>
</div>

<br/>

Are you pulling your hair out trying to run the `nvidia/LocateAnything-3B` model locally? 🤯 Getting strange hallucinated outputs like `!!!!!`, dealing with HuggingFace cache errors, or finding that the generation loops are completely broken on newer **NVIDIA Series 50 GPUs**? 

You are not alone! Many developers face compatibility issues when running this powerful Vision-Language Model on modern hardware. **But fear not!** 🦸‍♂️ This repository provides the exact fix and a flawless local implementation script (no Gradio required!). 

By pinning the precise dependencies and using our refactored script, you can generate pixel-perfect bounding box detections in minutes. 🚀

---

## 🛠️ Step-by-Step Setup Guide

Follow these steps exactly to guarantee a 100% clean and working environment:

### 1️⃣ Create a Clean Environment
First, ensure you have a pristine Conda environment using Python 3.12:
```bash
conda create -n locateanything python=3.12 -y
conda activate locateanything
```

### 2️⃣ Install PyTorch
Install PyTorch and TorchVision. *(Note: Adjust the index URL if you need a different CUDA version)*:
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3️⃣ Install Critical Dependencies 🚨
**THIS IS THE MAGIC FIX:** The model completely breaks on `transformers 5.x`. You **must** install the exact `4.57.1` version used during training/validation!
```bash
pip install transformers==4.57.1
```

### 4️⃣ Install Remaining Libraries
Grab the rest of the required packages for image processing and inference:
```bash
pip install opencv-python-headless Pillow numpy peft huggingface_hub decord lmdb
```

### 5️⃣ Run Inference! 🎉
Update `IMAGE_TO_PROCESS` and `TARGET_CATEGORIES` at the bottom of `main.py`, then run:
```bash
python main.py
```
*Voilà! The script will output an image named `[your_image]_detected.jpg` with beautiful, colorful bounding boxes drawn directly on it!* 🖼️✨
