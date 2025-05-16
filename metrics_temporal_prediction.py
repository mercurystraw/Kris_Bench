import os
import json
import base64
import time
import re
import logging
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# import the generic extractor that returns (score, reasoning)
from metrics_common import extract_score_and_reason
from utils.prompts import (
    prompt_consist_temporal,
    prompt_instruction_temporal,
    prompt_quality,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    default=["doubao", "gpt", "gemini"],
)
parser.add_argument("--category", type=str, default="count_change", help="category name")
args = parser.parse_args()

# Constants
BENCH_DIR = "KRIS_Bench"
RESULTS_DIR = "results"
MODELS = args.models
CATEGORIES   = ["temporal_prediction"]
METRICS      = ["consistency", "instruction_following", "image_quality"]

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    try:
        from PIL import Image
        import io
        
        # Open image and convert to JPEG format in memory
        img = Image.open(image_path)
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # Encode the JPEG data to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def evaluate_temporal_images(model_name, category, image_id, metrics=None):
    """
    Evaluate a single temporal‐prediction case:
    three reference frames + one predicted frame.
    Returns a dict mapping:
      metric -> score,
      metric_reasoning -> reasoning text.
    """
    if metrics is None:
        metrics = METRICS
    results = {}

    # load annotation
    annotation_path = os.path.join(BENCH_DIR, category, "annotation.json")
    try:
        with open(annotation_path, "r", encoding="utf-8") as af:
            annotations = json.load(af)
    except Exception as e:
        logging.error(f"Error loading annotation file {annotation_path}: {e}")
        return results

    ann = annotations.get(str(image_id))
    if not ann:
        logging.error(f"Image ID {image_id} not found in annotations for {category}")
        return results

    # build file paths
    refs = ann.get("ori_img", [])
    ref_paths = [os.path.join(BENCH_DIR, category, fn) for fn in refs]
    pred_path = os.path.join(RESULTS_DIR, model_name, category, f"{image_id}.jpg")

    # check existence
    for p in ref_paths:
        if not os.path.exists(p):
            logging.error(f"Reference image not found: {p}")
            return results
    if not os.path.exists(pred_path):
        logging.error(f"Predicted image not found: {pred_path}")
        return results

    # encode images
    ref_b64s = []
    for p in ref_paths:
        b64 = encode_image_to_base64(p)
        if not b64:
            logging.error(f"Failed to encode reference image: {p}")
            return results
        ref_b64s.append(b64)

    pred_b64 = encode_image_to_base64(pred_path)
    if not pred_b64:
        logging.error(f"Failed to encode predicted image: {pred_path}")
        return results

    instruction = ann.get("ins_en", "")
    
    # Determine the frame number of the predicted frame based on reference filenames
    # Extract frame numbers from reference filenames
    frame_numbers = []
    for ref in refs:
        match = re.search(r'(\d+)-(\d+)', ref)
        if match:
            frame_numbers.append(int(match.group(2)))
    
    # Determine the predicted frame number
    # Since we have 3 reference frames and need to predict 1 frame,
    # the predicted frame is the one missing from frames 1-4
    all_possible_frames = set([1, 2, 3, 4])
    existing_frames = set(frame_numbers)
    missing_frames = all_possible_frames - existing_frames
    
    # There should be exactly one missing frame
    if len(missing_frames) == 1:
        pred_frame_num = list(missing_frames)[0]
    else:
        # Default to next frame if we can't determine
        pred_frame_num = max(frame_numbers) + 1 if frame_numbers else 1

    # for each metric, get both score and reasoning
    for metric in metrics:
        if metric == "consistency":
            prompt = prompt_consist_temporal.format(N=pred_frame_num, instruct=instruction)
            resp = evaluate_temporal_with_gpt(prompt, ref_b64s, pred_b64, frame_numbers, pred_frame_num)
            score, reason = extract_score_and_reason(
                resp,
                score_key="consistency_score",
                reason_fields=["reasoning", "reason"],
            )
            results["consistency_score"] = score
            results["consistency_reasoning"] = reason
        elif metric == "instruction_following":
            prompt = prompt_instruction_temporal.format(N=pred_frame_num, instruct=instruction)
            resp = evaluate_temporal_with_gpt(prompt, ref_b64s, pred_b64, frame_numbers, pred_frame_num)
            score, reason = extract_score_and_reason(
                resp,
                score_key="instruction_score",
                reason_fields=["reasoning", "reason"],
            )
            results["instruction_score"] = score
            results["instruction_reasoning"] = reason
        elif metric == "image_quality":
            resp = evaluate_with_gpt(prompt_quality, None, pred_b64)
            score, reason = extract_score_and_reason(
                resp,
                score_key="quality_score",
                reason_fields=["reasoning", "reason"],
            )
            results["quality_score"] = score
            results["quality_reasoning"] = reason
        else:
            logging.warning(f"Unknown metric: {metric}")

    return results

def evaluate_temporal_with_gpt(prompt, reference_base64_list, predicted_base64, frame_numbers, pred_frame_num):
    """
    Send reference frames + predicted frame to GPT and return its response.
    Frames are sent in numerical order with the predicted frame inserted at its proper position.
    """
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt})
    
    # Create a combined list of all frames (reference + predicted)
    all_frames = []
    for i, b64 in enumerate(reference_base64_list):
        all_frames.append((frame_numbers[i], b64, "Reference"))
    
    # Add the predicted frame
    all_frames.append((pred_frame_num, predicted_base64, "Generated"))
    
    # Sort frames by frame number
    all_frames.sort(key=lambda x: x[0])
    
    # Add frames to the message in order
    for frame_num, b64, frame_type in all_frames:
        messages[0]["content"].append({"type": "text", "text": f"Frame {frame_num} ({frame_type}):"})
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=False,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"GPT call failed (attempt {attempt+1}/3): {e}")
            time.sleep(5)
    return ""

def evaluate_with_gpt(prompt, original_base64=None, edited_base64=None):
    """
    Send a single image to GPT for quality evaluation.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if edited_base64:
        messages[0]["content"].append({"type": "text", "text": "This is the image to evaluate:"})
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{edited_base64}"}
        })

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=False,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"GPT call failed (attempt {attempt+1}/3): {e}")
            time.sleep(5)
    return ""

def process_temporal_image_eval(model, category, image_id, metrics, annotations):
    """
    Thread worker: evaluate and package results for one image.
    """
    res = evaluate_temporal_images(model, category, image_id, metrics)
    if not res:
        return None
    ann = annotations.get(str(image_id), {})
    data = {
        "instruction": ann.get("ins_en", ""),
        "explain":     ann.get("explain_en", ""),
        **res
    }
    return image_id, data

def run_temporal_evaluation(models=None, categories=None, metrics=None, max_workers=8):
    """
    Master loop: loads existing metrics, dispatches workers, saves updates.
    """
    if models    is None: models    = MODELS
    if categories is None: categories = CATEGORIES
    if metrics   is None: metrics   = METRICS

    # mapping of metric to expected result keys
    expected_keys_map = {
        "consistency":          ["consistency_score"],
        "instruction_following": ["instruction_score"],
        "image_quality":        ["quality_score"],
    }

    for model in models:
        for category in tqdm(categories, desc=f"Evaluating {model}", unit="cat"):
            # load annotations
            ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
            try:
                with open(ann_file, "r", encoding="utf-8") as af:
                    annotations = json.load(af)
                    image_ids = sorted(annotations.keys())
            except Exception as e:
                logging.error(f"Failed loading annotations: {e}")
                continue

            # prepare metrics file
            out_dir     = os.path.join(RESULTS_DIR, model, category)
            os.makedirs(out_dir, exist_ok=True)
            metrics_fp  = os.path.join(out_dir, "metrics.json")
            
            # Load existing metrics if present
            try:
                if os.path.isfile(metrics_fp):
                    with open(metrics_fp, "r", encoding="utf-8") as mf:
                        metrics_data = json.load(mf)
                else:
                    metrics_data = {}
            except Exception as e:
                logging.error(f"Error loading existing metrics at {metrics_fp}: {e}")
                metrics_data = {}

            # Process all images regardless of whether they've been processed before
            to_process = image_ids

            if not to_process:
                logging.info(f"No images to process for {model}/{category}.")
                continue

            # dispatch workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                fut_map = {
                    executor.submit(process_temporal_image_eval, model, category, img_id, metrics, annotations): img_id
                    for img_id in to_process
                }
                for fut in tqdm(as_completed(fut_map), total=len(fut_map),
                                desc=f"{model}/{category}", leave=False):
                    img_id = fut_map[fut]
                    try:
                        out = fut.result()
                        if out:
                            _id, data = out
                            metrics_data[_id] = data
                    except Exception as e:
                        logging.error(f"Failed processing {img_id}: {e}")

            # save updated metrics
            try:
                with open(metrics_fp, "w", encoding="utf-8") as mf:
                    json.dump(metrics_data, mf, ensure_ascii=False, indent=2)
                logging.info(f"Saved metrics to {metrics_fp}")
            except Exception as e:
                logging.error(f"Failed to save metrics: {e}")

if __name__ == "__main__":
    run_temporal_evaluation(max_workers=15)