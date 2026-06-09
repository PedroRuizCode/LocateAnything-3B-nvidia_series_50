"""
LocateAnything-3B Inference Script
-----------------------------------
This script runs inference using the nvidia/LocateAnything-3B model.
It loads an image, prompts the model to locate specified objects, parses
the output bounding boxes, and draws them directly onto the image.
"""

import os
import re
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel, AutoTokenizer

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_color_for_label(label: str) -> tuple:
    """Generate a consistent RGB color for a given label string."""
    colors = [
        (8, 145, 178), (220, 38, 38), (22, 163, 74),
        (37, 99, 235), (217, 119, 6), (147, 51, 234)
    ]
    idx = sum(ord(c) for c in label)
    return colors[idx % len(colors)]


def parse_mixed_results(text: str, category_str: str = "") -> list:
    """
    Parse the raw output text containing <ref> and <box> tags into a list of dictionaries
    with bounding box coordinates and labels.
    """
    results = []
    expected_cats = [c.strip().lower() for c in category_str.split("</c>") if c.strip()]
    ref_box_pattern = r"(<ref>.*?</ref>)|(<box>.*?</box>)"
    current_label = None
    found_structured = False
    
    for m in re.finditer(ref_box_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        token = m.group(0)
        if token.lower().startswith("<ref>"):
            label_raw = re.sub(r"</?ref>", "", token, flags=re.IGNORECASE).strip()
            if label_raw:
                current_label = label_raw
        else:
            content = re.sub(r"</?box>", "", token, flags=re.IGNORECASE)
            nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
            coords = [float(n) for n in nums]
            if not coords:
                continue
            
            label = current_label if current_label is not None else (expected_cats[0] if expected_cats else "object")
            
            if len(coords) == 4:
                results.append({"type": "box", "coords": coords, "label": label})
            elif len(coords) == 2:
                results.append({"type": "point", "coords": coords, "label": label})
            found_structured = True
            
    if found_structured:
        return results

    # Fallback parsing if structure isn't perfect
    box_pattern = r"<box>(.*?)</box>"
    parts = re.split(box_pattern, text)
    for i in range(1, len(parts), 2):
        preceding_text = parts[i - 1].lower()
        content = parts[i]
        label = expected_cats[0] if expected_cats else "object"
        for cat in expected_cats:
            if cat in preceding_text:
                label = cat
                break
        
        nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
        coords = [float(n) for n in nums]
        if len(coords) == 4:
            results.append({"type": "box", "coords": coords, "label": label})
        elif len(coords) == 2:
            results.append({"type": "point", "coords": coords, "label": label})
            
    return results


def draw_on_frame(frame_bgr: np.ndarray, results: list, draw_label: bool = True) -> np.ndarray:
    """
    Draw bounding boxes and labels onto the image frame.
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_draw = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    w_img, h_img = pil_img.size
    parsed = []
    
    # Pre-calculate scaled coordinates
    for res in results:
        label = res.get("label", "object")
        color = get_color_for_label(label)
        c = res["coords"]
        
        if res.get("type") == "point":
            cx = max(0, min(w_img, c[0] * w_img / 1000))
            cy = max(0, min(h_img, c[1] * h_img / 1000))
            parsed.append(("point", label, color, cx, cy))
            continue
            
        if len(c) < 4:
            continue
            
        x1, y1, x2, y2 = c[0] * w_img / 1000, c[1] * h_img / 1000, c[2] * w_img / 1000, c[3] * h_img / 1000
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        parsed.append(("box", label, color, x1, y1, x2, y2))
    
    # Draw shapes
    for item in parsed:
        if item[0] == "box":
            _, _, color, x1, y1, x2, y2 = item
            draw.rectangle([x1, y1, x2, y2], fill=color + (65,), outline=color, width=4)
        elif item[0] == "point":
            _, _, color, cx, cy = item
            draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=color, outline="white", width=2)
            
    # Draw labels
    if draw_label:
        for item in parsed:
            if item[0] == "box":
                _, label, color, x1, y1, x2, y2 = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th, tw = t_box[3] - t_box[1], t_box[2] - t_box[0]
                pad_x, pad_y = 7, 4
                tag_y = y1 - (th + pad_y * 2) - 2
                if tag_y < 0:
                    tag_y = y2 + 2
                draw.rectangle([x1, tag_y, x1 + tw + pad_x * 2, tag_y + th + pad_y * 2], fill=color)
                draw.text((x1 + pad_x, tag_y + pad_y), label, fill="white", font=font)
                
    combined = Image.alpha_composite(img_draw, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def main(image_path: str, category_input: str, model_id: str = "nvidia/LocateAnything-3B", max_new_tokens: int = 512):
    """
    Main execution pipeline for loading the model, formatting input, running inference,
    and rendering bounding boxes on the output image.
    """
    # 1. Format user categories to model-expected XML style
    category_formatted = "</c>".join(c.strip() for c in category_input.split(",") if c.strip())
    query = f"Locate all the instances that matches the following description: {category_formatted}."
    
    # 2. Load model and processor
    print("Loading tokenizer + processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print("Loading model onto GPU...")
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to("cuda").eval()
    print("Model loaded ✓\n")
    
    # 3. Load image and build the conversation prompt
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }
    ]
    
    # 4. Preprocess inputs using the vision processor
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
    ).to("cuda")
    
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_hws = inputs.get("image_grid_hws", None)
    
    # 5. Run inference
    print("Running inference...")
    with torch.no_grad():
        result = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="slow",  # Standard AR generation
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=True,
        )
        
    # Unpack generation output
    if isinstance(result, tuple) and len(result) >= 3:
        output_text, token_sequence, out_info = result
    else:
        output_text = result
        out_info = ""
        
    print("\n── Raw model output ─────────────────────────────────────────────────")
    print(output_text)
    if out_info:
        print("\n── Stats ────────────────────────────────────────────────────────────")
        print(out_info)
        
    # 6. Parse results and generate bounding box visualization
    print("\n── Processing and Saving Image ──────────────────────────────────────")
    detections = parse_mixed_results(output_text, category_formatted)
    
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    out_img_bgr = draw_on_frame(frame_bgr, detections, draw_label=True)
    
    # Generate dynamic output filename based on original image name
    ori_img_name = os.path.basename(image_path).split(".")[0]
    out_path = f"{ori_img_name}_detected.jpg"
    
    cv2.imwrite(out_path, out_img_bgr)
    print(f"Saved {len(detections)} detections to {out_path}")


if __name__ == "__main__":
    # Define execution parameters
    IMAGE_TO_PROCESS = "./test1.jpg"
    TARGET_CATEGORIES = "Aircraft, Car, Hangar"
    
    main(image_path=IMAGE_TO_PROCESS, category_input=TARGET_CATEGORIES)