import os
import json
import base64
import time
import logging
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.prompts import (
    prompt_consist,
    prompt_quality,
    prompt_view_instruction_following,
    prompt_instruction_following
)
from metrics_common import (
    extract_consistency_score_and_reason,
    extract_instruction_score_and_reason,
    extract_quality_score_and_reason
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Constants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    default=["doubao", "gpt", "gemini"],
)
args = parser.parse_args()

# Constants
BENCH_DIR = "KRIS_Bench"
RESULTS_DIR = "results"
MODELS = args.models
CATEGORIES = ["viewpoint_change"]
METRICS = ["consistency", "instruction_following", "image_quality"]

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.error("Error encoding image %s: %s", image_path, e)
        return None

def evaluate_images(model_name, category, image_id, metrics=None):
    """
    Evaluate images for a specific model, category and image ID.
    Returns a dict with score and reasoning for each metric.
    """
    metrics = metrics or METRICS
    results = {}

    annotation_path = os.path.join(BENCH_DIR, category, "annotation.json")
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except Exception as e:
        logging.error("Failed to load annotation file %s: %s", annotation_path, e)
        return results

    ann = annotations.get(str(image_id))
    if not ann:
        logging.warning("Image ID %s not found in %s", image_id, annotation_path)
        return results

    # Build paths
    ori_path = os.path.join(BENCH_DIR, category, ann["ori_img"])
    edited_path = os.path.join(RESULTS_DIR, model_name, category, f"{image_id}.jpg")
    gt_rel = ann.get("gt_img")
    gt_path = os.path.join(BENCH_DIR, category, gt_rel) if gt_rel else None

    # Validate files
    if not os.path.exists(ori_path):
        logging.error("Original image missing: %s", ori_path)
        return results
    if not os.path.exists(edited_path):
        logging.error("Edited image missing: %s", edited_path)
        return results
    if category == "viewpoint_change" and (not gt_path or not os.path.exists(gt_path)):
        logging.error("Ground-truth image missing: %s", gt_path)
        return results

    # Encode images
    ori_b64 = encode_image_to_base64(ori_path)
    edt_b64 = encode_image_to_base64(edited_path)
    gt_b64 = encode_image_to_base64(gt_path) if gt_path else None
    if not ori_b64 or not edt_b64 or (category == "viewpoint_change" and not gt_b64):
        logging.error("Failed to encode images for %s/%s/%s", model_name, category, image_id)
        return results

    instr = ann.get("ins_en", "")
    for metric in metrics:
        if metric == "consistency":
            prompt = prompt_consist.format(instruct=instr)
            resp = evaluate_with_gpt(prompt, ori_b64, edt_b64)
            score, reason = extract_consistency_score_and_reason(resp)
            results["consistency_score"] = score
            results["consistency_reasoning"] = reason
        elif metric == "instruction_following":
            if category == "viewpoint_change":
                prompt = prompt_view_instruction_following.format(instruct=instr)
                resp = evaluate_with_gpt(prompt, ori_b64, edt_b64, gt_b64)
            else:
                prompt = prompt_instruction_following.format(instruct=instr)
                resp = evaluate_with_gpt(prompt, ori_b64, edt_b64)
            score, reason = extract_instruction_score_and_reason(resp)
            results["instruction_score"] = score
            results["instruction_reasoning"] = reason
        elif metric == "image_quality":
            resp = evaluate_with_gpt(prompt_quality, None, edt_b64)
            score, reason = extract_quality_score_and_reason(resp)
            results["quality_score"] = score
            results["quality_reasoning"] = reason
        else:
            logging.warning("Unknown metric: %s", metric)

    return results

def evaluate_with_gpt(prompt, original_b64=None, edited_b64=None, gt_b64=None):
    """Call GPT with images/text and retry on failure."""
    content = [{"type": "text", "text": prompt}]
    if original_b64:
        content += [
            {"type": "text", "text": "This is the original image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}}
        ]
    if edited_b64:
        content += [
            {"type": "text", "text": "This is the edited image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_b64}"}}
        ]
    if gt_b64:
        content += [
            {"type": "text", "text": "This is the ground truth image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}}
        ]

    messages = [{"role": "user", "content": content}]
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
            logging.error("GPT call failed (attempt %d): %s", attempt+1, e)
            time.sleep(5)
    logging.error("Failed to get evaluation from GPT after retries")
    return ""

def process_image_eval(model, category, image_id, metrics, annotations):
    """Helper for threaded evaluation of a single image."""
    results = evaluate_images(model, category, image_id, metrics)
    if not results:
        return None
    ann = annotations.get(image_id, {})
    data = {
        "instruction": ann.get("ins_en", ""),
        "explain": ann.get("explain_en", ""),
        **results
    }
    return image_id, data

def run_evaluation(models=None, categories=None, metrics=None, max_workers=8):
    """
    Run evaluation for specified models, categories and metrics using multithreading.
    """
    models = models or MODELS
    categories = categories or CATEGORIES
    metrics = metrics or METRICS
    
    # mapping of metric to expected result keys
    expected_keys_map = {
        "consistency":          ["consistency_score"],
        "instruction_following": ["instruction_score"],
        "image_quality":        ["quality_score"],
    }

    for model in models:
        for category in tqdm(categories, desc=f"Evaluating {model}"):
            ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            except Exception as e:
                logging.error("Error loading annotation %s: %s", ann_file, e)
                continue

            out_dir = os.path.join(RESULTS_DIR, model, category)
            os.makedirs(out_dir, exist_ok=True)
            metrics_file = os.path.join(out_dir, "metrics.json")
            
            # Load existing metrics if present
            try:
                if os.path.isfile(metrics_file):
                    with open(metrics_file, "r", encoding="utf-8") as mf:
                        metrics_data = json.load(mf)
                else:
                    metrics_data = {}
            except Exception as e:
                logging.error(f"Error loading existing metrics at {metrics_file}: {e}")
                metrics_data = {}
                
            # Determine which images need evaluation
            to_process = []
            for img_id in annotations:
                rec = metrics_data.get(img_id)
                if rec:
                    complete = True
                    for m in metrics:
                        for key in expected_keys_map.get(m, []):
                            if key not in rec or rec[key] is None:
                                complete = False
                                break
                        if not complete:
                            break
                    if complete:
                        continue
                to_process.append(img_id)
                
            if not to_process:
                logging.info(f"No images to process for {model}/{category}.")
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_image_eval, model, category, img_id, metrics, annotations): img_id
                    for img_id in to_process
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"{model}/{category}", leave=False):
                    res = fut.result()
                    if res:
                        img_id, data = res
                        logging.debug("Result %s: %s", img_id, data)
                        metrics_data[img_id] = data

            try:
                with open(metrics_file, "w", encoding="utf-8") as wf:
                    json.dump(metrics_data, wf, ensure_ascii=False, indent=2)
                logging.info("Saved metrics to %s", metrics_file)
            except Exception as e:
                logging.error("Failed to save metrics to %s: %s", metrics_file, e)

if __name__ == "__main__":
    run_evaluation(max_workers=15)
