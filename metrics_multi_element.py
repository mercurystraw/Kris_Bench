import os
import json
import base64
import time
import re
import logging
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from metrics_common import extract_score_and_reason
from utils.prompts import (
    prompt_consist_multi,
    prompt_instruction_multi,
    prompt_quality,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
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
CATEGORIES = ["multi-element_composition"]
METRICS = ["consistency", "instruction_following", "image_quality"]

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image_to_base64(image_path):
    """Encode an image file to a base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def evaluate_multi_element_images(model_name, category, image_id, metrics=None):
    """
    Evaluate one multi-element synthesis example (3 refs + 1 prediction).
    Returns a dict containing both score and reasoning for each metric.
    """
    if metrics is None:
        metrics = METRICS

    # load annotations
    ann_path = os.path.join(BENCH_DIR, category, "annotation.json")
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load annotation file {ann_path}: {e}")
        return {}

    ann = annotations.get(str(image_id))
    if not ann:
        logging.error(f"Image ID {image_id} not in annotations for {category}")
        return {}

    # build file paths
    ref_names = ann.get("ori_img", [])
    ref_paths = [os.path.join(BENCH_DIR, category, n) for n in ref_names]
    pred_path = os.path.join(RESULTS_DIR, model_name, category, f"{image_id}.jpg")

    # verify existence
    for p in ref_paths:
        if not os.path.exists(p):
            logging.error(f"Reference image not found: {p}")
            return {}
    if not os.path.exists(pred_path):
        logging.error(f"Predicted image not found: {pred_path}")
        return {}

    # encode to base64
    ref_b64_list = []
    for p in ref_paths:
        b = encode_image_to_base64(p)
        if not b:
            logging.error(f"Failed to encode reference image: {p}")
            return {}
        ref_b64_list.append(b)
    pred_b64 = encode_image_to_base64(pred_path)
    if not pred_b64:
        logging.error(f"Failed to encode predicted image: {pred_path}")
        return {}

    instruction = ann.get("instruction", "")
    results = {}

    for metric in metrics:
        if metric == "consistency":
            prompt = prompt_consist_multi.format(instruct=instruction)
            resp = evaluate_multi_element_with_gpt(prompt, ref_b64_list, pred_b64)
            score, reason = extract_score_and_reason(
                resp,
                score_key="consistency_score",
                reason_fields=["consistency_reasoning"]
            )
            results["consistency_score"] = score
            results["consistency_reasoning"] = reason

        elif metric == "instruction_following":
            prompt = prompt_instruction_multi.format(instruct=instruction)
            resp = evaluate_multi_element_with_gpt(prompt, ref_b64_list, pred_b64)
            score, reason = extract_score_and_reason(
                resp,
                score_key="instruction_score",
                reason_fields=["instruction_reasoning"]
            )
            results["instruction_score"] = score
            results["instruction_reasoning"] = reason

        elif metric == "image_quality":
            resp = evaluate_with_gpt(prompt_quality, edited_base64=pred_b64)
            score, reason = extract_score_and_reason(
                resp,
                score_key="quality_score",
                reason_fields=["quality_reasoning"]
            )
            results["quality_score"] = score
            results["quality_reasoning"] = reason

        else:
            logging.warning(f"Unknown metric: {metric}")

    return results

def evaluate_multi_element_with_gpt(prompt, reference_base64_list, predicted_base64):
    """Call GPT with 3 reference images and 1 predicted image in one shot."""
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt})

    for idx, ref in enumerate(reference_base64_list, start=1):
        messages[0]["content"].append({"type": "text", "text": f"Reference Image {idx}:"})
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{ref}"}
        })

    messages[0]["content"].append({"type": "text", "text": "Predicted Image:"})
    messages[0]["content"].append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{predicted_base64}"}
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
            logging.error(f"GPT error (attempt {attempt+1}/3): {e}")
            time.sleep(5)
    logging.error("Failed to get evaluation from GPT after 3 attempts")
    return ""

def evaluate_with_gpt(prompt, original_base64=None, edited_base64=None):
    """Call GPT for single-image evaluation (image_quality)."""
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt})
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
            logging.error(f"GPT error (attempt {attempt+1}/3): {e}")
            time.sleep(5)
    logging.error("Failed to get evaluation from GPT after 3 attempts")
    return ""

def process_multi_element_image_eval(model, category, image_id, metrics, annotations):
    """Thread worker: evaluate one image and package results."""
    eval_res = evaluate_multi_element_images(model, category, image_id, metrics)
    if not eval_res:
        return None
    ann = annotations[image_id]
    packed = {
        "instruction": ann.get("ins_en", ""),
        "explain": ann.get("explain_en", ""),
        **eval_res
    }
    return image_id, packed

def run_multi_element_evaluation(models=None, categories=None, metrics=None, max_workers=8):
    """
    Mirror common.run_evaluation: load existing metrics, dispatch threads, save results.
    """
    if models is None:
        models = MODELS
    if categories is None:
        categories = CATEGORIES
    if metrics is None:
        metrics = METRICS

    # mapping of metric to expected result keys
    expected_keys_map = {
        "consistency":          ["consistency_score"],
        "instruction_following": ["instruction_score"],
        "image_quality":        ["quality_score"],
    }

    for model in models:
        for category in categories:
            logging.info(f"Start {model}/{category}")
            ann_path = os.path.join(BENCH_DIR, category, "annotation.json")
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            except Exception as e:
                logging.error(f"Cannot load annotations {ann_path}: {e}")
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

            logging.info(f"{model}/{category}: {len(to_process)} / {len(annotations)} to process")
            
            if not to_process:
                logging.info(f"No images to process for {model}/{category}.")
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {
                    exe.submit(process_multi_element_image_eval, model, category, img, metrics, annotations): img
                    for img in to_process
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"{model}/{category}", leave=False):
                    img = futures[fut]
                    try:
                        res = fut.result()
                        if res:
                            img_id, data = res
                            metrics_data[img_id] = data
                    except Exception as e:
                        logging.error(f"Failed processing {img}: {e}")

            try:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved metrics to {metrics_file}")
            except Exception as e:
                logging.error(f"Failed to save metrics to {metrics_file}: {e}")

if __name__ == "__main__":
    run_multi_element_evaluation(max_workers=15)