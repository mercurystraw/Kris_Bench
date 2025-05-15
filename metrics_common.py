import os
import json
import base64
import time
import re
import logging
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ReasonEdit.prompts import (
    prompt_consist,
    prompt_quality,
    prompt_instruction_following,
    prompt_abnormal_instruction_following,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# Suppress verbose HTTP logs from OpenAI/httpx
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
args = parser.parse_args()

# Constants
BENCH_DIR = "KRIS_Bench"
RESULTS_DIR = "results"
MODELS = args.models
CATEGORIES = [
    "object_count", "color_change", "anomaly_correction",
    "position_movement", "size_adjustment", "part_completion",
    "multi-instruction_execution",
]
METRICS = ["consistency", "instruction_following", "image_quality"]

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image_to_base64(path: str) -> str | None:
    """Read an image file and return its base64-encoded string (or None on error)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to encode image {path}: {e}")
        return None

def extract_json_field(response: str, score_key: str, reason_key: str) -> tuple[int | None, str | None]:
    """Try to parse a single-key JSON object out of the response."""
    pattern = r"\{[^{}]*" + re.escape(score_key) + r"[^{}]*\}"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None, None
    try:
        data = json.loads(match.group(0))
        score = data.get(score_key)
        reason = data.get(reason_key)
        if score is not None:
            score = int(score)
        return score, reason
    except Exception:
        return None, None

# default regex patterns for matching "n/5", "n out of 5", or standalone digit
DEFAULT_PATTERNS = [
    r"([1-5])\s*/\s*5",
    r"([1-5])\s+out\s+of\s+5",
    r"\b([1-5])\b",
]

def extract_score_and_reason(response: str,
                             score_key: str,
                             reason_fields: list[str],
                             prefix_patterns: list[str] | None = None) -> tuple[int | None, str | None]:
    """
    Generic extractor for a score/reason pair. Tries JSON first,
    then falls back to regex patterns.
    """
    # JSON extraction
    for rf in reason_fields:
        score, reason = extract_json_field(response, score_key, rf)
        if score is not None:
            return score, reason
    # regex fallback
    patterns = (prefix_patterns or []) + DEFAULT_PATTERNS
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            return int(m.group(1)), None
    return None, None

def extract_consistency_score_and_reason(response: str) -> tuple[int | None, str | None]:
    return extract_score_and_reason(
        response,
        score_key="consistency_score",
        reason_fields=["reason", "reasoning"],
        prefix_patterns=[r"consistency[_\s]*score\s*[:：]?\s*([1-5])"]
    )

def extract_instruction_score_and_reason(response: str) -> tuple[int | None, str | None]:
    return extract_score_and_reason(
        response,
        score_key="instruction_score",
        reason_fields=["reasoning", "reason"],
        prefix_patterns=[r"instruction[_\s]*score\s*[:：]?\s*([1-5])"]
    )

def extract_quality_score_and_reason(response: str) -> tuple[int | None, str | None]:
    return extract_score_and_reason(
        response,
        score_key="quality_score",
        reason_fields=["reasoning", "reason"],
        prefix_patterns=[r"quality[_\s]*score\s*[:：]?\s*([1-5])"]
    )

def evaluate_with_gpt(prompt: str,
                      original_base64: str | None = None,
                      edited_base64: str | None = None) -> str:
    """
    Send a chat completion request with text+image inputs, retrying up to 3 times.
    Returns the assistant's content or empty string on failure.
    """
    message = {"role": "user", "content": []}
    message["content"].append({"type": "text", "text": prompt})

    if original_base64:
        message["content"].append({"type": "text", "text": "This is the original image:"})
        message["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{original_base64}"}
        })
    if edited_base64:
        message["content"].append({"type": "text", "text": "This is the edited image:"})
        message["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{edited_base64}"}
        })

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[message],
                stream=False,
                max_tokens=1000
            )
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"GPT call failed (attempt {attempt+1}/3): {e}")
            time.sleep(5)
    logging.error("GPT evaluation failed after 3 attempts.")
    return ""

def evaluate_images(model_name: str,
                    category: str,
                    image_id: str,
                    metrics: list[str] | None = None) -> dict:
    """
    Load annotation + images, run the specified metrics, and return a dict
    with keys: 'consistency_score', 'consistency_reasoning',
               'instruction_score', 'instruction_reasoning',
               'quality_score', 'quality_reasoning'
    (empty if errors).
    """
    metrics = metrics or METRICS
    results: dict = {}

    ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
    if not os.path.isfile(ann_file):
        logging.error(f"Missing annotation.json at {ann_file}")
        return results

    try:
        with open(ann_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except Exception as e:
        logging.error(f"Error loading annotation file {ann_file}: {e}")
        return results

    entry = annotations.get(str(image_id))
    if entry is None:
        logging.error(f"No annotation for {category}/{image_id}")
        return results

    orig_path = os.path.join(BENCH_DIR, category, entry["ori_img"])
    edit_path = os.path.join(RESULTS_DIR, model_name, category, f"{image_id}.jpg")
    if not os.path.isfile(orig_path) or not os.path.isfile(edit_path):
        logging.error(f"Missing image file(s): {orig_path}, {edit_path}")
        return results

    orig_b64 = encode_image_to_base64(orig_path)
    edit_b64 = encode_image_to_base64(edit_path)
    if orig_b64 is None or edit_b64 is None:
        logging.error(f"Failed to encode images for {model_name}/{category}/{image_id}")
        return results

    instr = entry.get("instruction", "")
    expl = entry.get("explain", "")

    for m in metrics:
        if m == "consistency":
            prompt = prompt_consist.format(instruct=instr)
            resp = evaluate_with_gpt(prompt, original_base64=orig_b64, edited_base64=edit_b64)
            score, reasoning = extract_consistency_score_and_reason(resp)
            results["consistency_score"] = score
            results["consistency_reasoning"] = reasoning

        elif m == "instruction_following":
            if "abnormality_correction" in category:
                prompt = prompt_abnormal_instruction_following.format(instruct=instr, explanation=expl)
            else:
                prompt = prompt_instruction_following.format(instruct=instr)
            resp = evaluate_with_gpt(prompt, original_base64=orig_b64, edited_base64=edit_b64)
            score, reasoning = extract_instruction_score_and_reason(resp)
            results["instruction_score"] = score
            results["instruction_reasoning"] = reasoning

        elif m == "image_quality":
            resp = evaluate_with_gpt(prompt_quality, edited_base64=edit_b64)
            score, reasoning = extract_quality_score_and_reason(resp)
            results["quality_score"] = score
            results["quality_reasoning"] = reasoning

        else:
            logging.warning(f"Unknown metric '{m}'")
    return results

def process_image_eval(model: str,
                       category: str,
                       image_id: str,
                       metrics: list[str],
                       annotations: dict) -> tuple[str, dict] | None:
    """
    Thread worker: evaluate one image, return (image_id, data_dict) or None.
    """
    eval_res = evaluate_images(model, category, image_id, metrics)
    if not eval_res:
        return None
    entry = annotations.get(str(image_id), {})
    data = {
        "instruction": entry.get("instruction", ""),
        "explain":     entry.get("explain", ""),
        **eval_res
    }
    return image_id, data

def run_evaluation(models: list[str] | None = None,
                   categories: list[str] | None = None,
                   metrics: list[str] | None = None,
                   max_workers: int = 8) -> None:
    """
    Run image evaluations across models, categories, and metrics with multithreading.
    If an existing metrics.json has entries for an image, check if all required
    scores are present; if not, re-run evaluation for that image.
    """
    models     = models     or MODELS
    categories = categories or CATEGORIES
    metrics    = metrics    or METRICS

    # mapping of metric to expected result keys
    expected_keys_map = {
        "consistency":          ["consistency_score"],
        "instruction_following": ["instruction_score"],
        "image_quality":        ["quality_score"],
    }

    for model in models:
        for category in tqdm(categories, desc=f"Evaluating {model}"):
            ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
            if not os.path.isfile(ann_file):
                logging.error(f"Missing annotation.json at {ann_file}")
                continue

            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            except Exception as e:
                logging.error(f"Error loading annotations at {ann_file}: {e}")
                continue

            image_ids = list(annotations.keys())
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

            # Determine which images need evaluation
            to_process: list[str] = []
            for img_id in image_ids:
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

            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_image_eval, model, category, img_id, metrics, annotations): img_id
                    for img_id in to_process
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc=f"{model}/{category}",
                                leave=False):
                    try:
                        res = fut.result()
                        if res:
                            img_id, data = res
                            metrics_data[img_id] = data
                    except Exception as e:
                        logging.error(f"Failed processing {futures[fut]}: {e}")

            # Save updated metrics
            try:
                with open(metrics_file, "w", encoding="utf-8") as wf:
                    json.dump(metrics_data, wf, ensure_ascii=False, indent=2)
                logging.info(f"Saved metrics to {metrics_file}")
            except Exception as e:
                logging.error(f"Failed to save metrics to {metrics_file}: {e}")

if __name__ == "__main__":
    run_evaluation(max_workers=15)
