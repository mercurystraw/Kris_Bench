import os
import json
import base64
import re
import time
import logging
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.prompts import (
    prompt_consist,
    prompt_dual_evaluation,
    prompt_quality,
)

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
CATEGORIES = [
    'abstract_reasoning', 'mathematics', 'practical_knowledge', 'medicine', 'rule-based_reasoning',
    'biology', 'geography', 'chemistry', 'humanities', 'physics',
]
METRICS = ["consistency", "dual_score", "image_quality"]

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image_to_base64(path: str) -> str | None:
    """Read an image file and return its base64-encoded string (or None on error)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to encode image {path}: {e}")
        return None

def evaluate_images(model: str, category: str, image_id: str, metrics: list[str] | None = None) -> dict:
    """
    Evaluate a single image on specified metrics and return the results dict,
    including both scores and reasoning extracted from the GPT responses.
    """
    metrics = metrics or METRICS
    results: dict = {}

    ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
    if not os.path.isfile(ann_file):
        logging.error(f"Annotation file not found: {ann_file}")
        return results

    try:
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except Exception as e:
        logging.error(f"Error loading annotation file {ann_file}: {e}")
        return results

    ann = annotations.get(str(image_id))
    if not ann:
        logging.warning(f"Image ID {image_id} not in annotations for category {category}")
        return results

    original_path = os.path.join(BENCH_DIR, category, ann["ori_img"])
    edited_path = os.path.join(RESULTS_DIR, model, category, f"{image_id}.jpg")

    if not os.path.isfile(original_path):
        logging.error(f"Original image not found: {original_path}")
        return results
    if not os.path.isfile(edited_path):
        logging.error(f"Edited image not found: {edited_path}")
        return results

    orig_b64 = encode_image_to_base64(original_path)
    edit_b64 = encode_image_to_base64(edited_path)
    if not orig_b64 or not edit_b64:
        logging.error(f"Failed to encode images for {model}/{category}/{image_id}")
        return results

    instruct = ann.get("ins_en", "")
    explain = ann.get("explain_en", "")

    for metric in metrics:
        if metric == "consistency":
            prompt = prompt_consist.format(instruct=instruct)
            resp = evaluate_with_gpt(prompt, orig_b64, edit_b64)
            results["consistency_score"]     = extract_consistency_score(resp)
            results["consistency_reasoning"] = extract_consistency_reasoning(resp)

        elif metric == "dual_score":
            prompt = prompt_dual_evaluation.format(instruct=instruct, explanation=explain)
            resp = evaluate_with_gpt(prompt, orig_b64, edit_b64)
            dual = extract_dual_scores(resp)
            results["instruction_score"]     = dual.get("instruction_score")
            results["instruction_reasoning"] = dual.get("instruction_reasoning")
            results["knowledge_score"]       = dual.get("knowledge_score")
            results["knowledge_reasoning"]   = dual.get("knowledge_reasoning")

        elif metric == "image_quality":
            resp = evaluate_with_gpt(prompt_quality, None, edit_b64)
            results["quality_score"]     = extract_quality_score(resp)
            results["quality_reasoning"] = extract_quality_reasoning(resp)

        else:
            logging.warning(f"Unknown metric: {metric}")

    return results

def evaluate_with_gpt(prompt: str,
                      original_b64: str | None = None,
                      edited_b64: str | None = None) -> str:
    """
    Send a chat completion request with prompt and optional images, with retry logic.
    Returns GPT response content or empty string on failure.
    """
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt})

    if original_b64:
        messages[0]["content"].extend([
            {"type": "text", "text": "This is the original image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}}
        ])

    if edited_b64:
        messages[0]["content"].extend([
            {"type": "text", "text": "This is the edited image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_b64}"}}
        ])

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
            logging.error(f"Error calling GPT (attempt {attempt+1}/3): {e}")
            time.sleep(5 if attempt < 2 else 0)

    logging.error("Failed to get evaluation from GPT after retries.")
    return ""

def extract_dual_scores(response: str) -> dict:
    """
    Extract instruction_score, instruction_reasoning,
    knowledge_score, and knowledge_reasoning from GPT response.
    """
    # Try JSON block first
    json_match = re.search(r"\{[^{}]*instruction_score[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {
                "instruction_score":    int(data.get("instruction_score", 0)) if data.get("instruction_score") is not None else None,
                "knowledge_score":      int(data.get("knowledge_score", 0))   if data.get("knowledge_score")   is not None else None,
                "instruction_reasoning": data.get("instruction_reasoning"),
                "knowledge_reasoning":   data.get("knowledge_reasoning"),
            }
        except Exception:
            logging.debug("Failed to parse JSON dual results; falling back to regex.")

    # Fallback regex parsing
    instr = knowl = None
    m1 = re.search(r"instruction[_\s]*score\s*[:：]\s*([1-5])", response, re.IGNORECASE)
    if m1:
        instr = int(m1.group(1))
    m2 = re.search(r"knowledge[_\s]*score\s*[:：]\s*([1-5])", response, re.IGNORECASE)
    if m2:
        knowl = int(m2.group(1))

    inst_reason = None
    know_reason = None
    m_ir = re.search(r"instruction_reasoning\s*[:：]\s*(.+?)(?=knowledge_reasoning|$)", response, re.DOTALL|re.IGNORECASE)
    if m_ir:
        inst_reason = m_ir.group(1).strip()
    m_kr = re.search(r"knowledge_reasoning\s*[:：]\s*(.+)", response, re.DOTALL|re.IGNORECASE)
    if m_kr:
        know_reason = m_kr.group(1).strip()

    return {
        "instruction_score":    instr,
        "knowledge_score":      knowl,
        "instruction_reasoning": inst_reason,
        "knowledge_reasoning":   know_reason,
    }

def extract_consistency_score(response: str) -> int | None:
    """
    Extract a consistency score (1-5) from GPT response.
    """
    json_match = re.search(r"\{[^{}]*consistency_score[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return int(data.get("consistency_score", 0)) if data.get("consistency_score") is not None else None
        except Exception:
            logging.debug("Failed to parse JSON consistency score.")

    patterns = [
        r"consistency[_\s]*score\s*[:：]\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return int(m.group(1))

    digits = re.findall(r"\b([1-5])\b", response)
    return int(digits[0]) if digits else None

def extract_consistency_reasoning(response: str) -> str | None:
    """
    Extract the reasoning text from the consistency evaluation response.
    """
    json_match = re.search(r"\{[^{}]*reasoning[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("reasoning")
        except Exception:
            logging.debug("Failed to parse JSON consistency reasoning.")
    m = re.search(r"reasoning\s*[:：]\s*(.+)", response, re.IGNORECASE|re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def extract_quality_score(response: str) -> int | None:
    """
    Extract an image quality score (1-5) from GPT response.
    """
    json_match = re.search(r"\{[^{}]*quality_score[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return int(data.get("quality_score", 0)) if data.get("quality_score") is not None else None
        except Exception:
            logging.debug("Failed to parse JSON quality score.")

    patterns = [
        r"quality[_\s]*score\s*[:：]\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return int(m.group(1))

    digits = re.findall(r"\b([1-5])\b", response)
    return int(digits[0]) if digits else None

def extract_quality_reasoning(response: str) -> str | None:
    """
    Extract the reasoning text from the image quality evaluation response.
    """
    json_match = re.search(r"\{[^{}]*reasoning[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("reasoning")
        except Exception:
            logging.debug("Failed to parse JSON quality reasoning.")
    m = re.search(r"reasoning\s*[:：]\s*(.+)", response, re.IGNORECASE|re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def process_image_eval(model: str,
                       category: str,
                       image_id: str,
                       metrics: list[str],
                       annotations: dict) -> tuple[str, dict] | None:
    """
    Thread worker: evaluate a single image and package metrics with annotations.
    """
    eval_res = evaluate_images(model, category, image_id, metrics)
    if not eval_res:
        return None

    ann = annotations.get(str(image_id), {})
    record = {
        "instruction": ann.get("ins_en", ""),
        "explain":     ann.get("explain_en", ""),
        **eval_res
    }
    return (image_id, record)

def run_evaluation(models: list[str] | None = None,
                   categories: list[str] | None = None,
                   metrics: list[str] | None = None,
                   max_workers: int = 8) -> None:
    """
    Run image evaluations across models, categories, metrics with multithreading.
    If an existing metrics.json has entries for an image, check if all required
    scores are present; if not, re-run evaluation for that image.
    """
    models     = models     or MODELS
    categories = categories or CATEGORIES
    metrics    = metrics    or METRICS

    # mapping of metric to expected result keys
    expected_keys_map = {
        "consistency":  ["consistency_score"],
        "dual_score":   ["instruction_score", "knowledge_score"],
        "image_quality": ["quality_score"],
    }

    for model in models:
        for category in tqdm(categories, desc=f"Evaluating {model}"):
            ann_file = os.path.join(BENCH_DIR, category, "annotation.json")
            if not os.path.isfile(ann_file):
                logging.error(f"Missing annotation.json: {ann_file}")
                continue

            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                image_ids = list(annotations.keys())
            except Exception as e:
                logging.error(f"Error reading annotations {ann_file}: {e}")
                continue

            out_dir = os.path.join(RESULTS_DIR, model, category)
            os.makedirs(out_dir, exist_ok=True)
            metrics_path = os.path.join(out_dir, "metrics.json")

            # Load existing metrics if present
            try:
                if os.path.isfile(metrics_path):
                    with open(metrics_path, 'r', encoding='utf-8') as rf:
                        metrics_data = json.load(rf)
                else:
                    metrics_data = {}
            except Exception as e:
                logging.error(f"Error loading existing metrics {metrics_path}: {e}")
                metrics_data = {}

            # Determine which images need evaluation
            to_process = []
            for img_id in image_ids:
                rec = metrics_data.get(img_id)
                if rec:
                    # check if all expected keys exist and are not None
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
                    res = fut.result()
                    if res:
                        img_id, data = res
                        metrics_data[img_id] = data

            # Save updated metrics
            try:
                with open(metrics_path, 'w', encoding='utf-8') as wf:
                    json.dump(metrics_data, wf, ensure_ascii=False, indent=2)
                logging.info(f"Saved metrics to {metrics_path}")
            except Exception as e:
                logging.error(f"Failed to save metrics to {metrics_path}: {e}")

if __name__ == "__main__":
    run_evaluation(max_workers=15)
