# KRIS-Bench Evaluation

This directory contains automated evaluation scripts for the KRIS-Bench.

## Directory Structure

- `metrics_common.py`: Evaluation script for tasks without knowledge plausibility.
- `metrics_view_change.py`: Evaluation script for Viewpoint Change task with ground truth image.
- `metrics_knowldge.py`: Evaluation script for tasks with knowledge plausibility.
- `metrics_multi_element.py`  Evaluation script for multi-elment composition task.
- `metrics_temporal_prediction.py` Evaluation script for temporal prediction tasks.
- `results/`: Directory for editing and evaluation results.
- `KRIS_Bench/`: Benchmark dataset directory, should contain `annotation.json` and original images for each category.

## Requirements
- Python 3.8+

Set your OpenAI API Key as an environment variable before running:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Evaluation Metrics
- Visual Consistency
- Visual Quality
- Instruction Following
- Knowledge Plausibility

## Usage

First, download the benchmark. We upload the whole benchmark to the [Hugging Face repository](https://huggingface.co/datasets/Liang0223/KRIS_Bench). For convenience, we also keep the benchmark in this repository. You can find the dataset in [KRIS_Bench](./KRIS_Bench).

Run the main evaluation script from the command line:

```bash
python metrics_common.py --models doubao gpt gemini
```

Arguments:

- `--models`: List of model names to evaluate

The script will automatically iterate over the specified models and categories, call GPT-4o for automated evaluation, and save results to `results/{model}/{category}/metrics.json`.

## Output Format

Each category will have a `metrics.json` file with the following structure:

```json
{
  "1": {
    "instruction": "...",
    "explain": "...",
    "consistency_score": 5,
    "consistency_reasoning": "...",
    "instruction_score": 5,
    "instruction_reasoning": "...",
    "quality_score": 4,
    "quality_reasoning": "..."
  },
  ...
}
```

## Notes

- Ensure `KRIS_Bench/{category}/annotation.json` and original images exist.
- Ensure model-generated images are present in `results/{model}/{category}/` and named as `{image_id}.jpg`.
- OpenAI API usage is subject to rate limits and costs. Adjust `max_workers` and batch size as needed.
