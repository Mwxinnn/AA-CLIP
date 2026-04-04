import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main(args):
    random.seed(args.seed)

    full_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(full_jsonl)

    # group by class and label
    by_class = defaultdict(lambda: {0: [], 1: []})
    for item in data:
        cls = item["class_name"]
        label = int(item["label"])
        by_class[cls][label].append(item)

    shot = args.shot  # x-shot = x normal + x anomaly
    fewshot_data = []

    for cls, samples in by_class.items():
        normals = samples[0]
        anomalies = samples[1]

        num_normal = min(len(normals), shot)
        num_anomaly = min(len(anomalies), shot)

        if num_normal < shot or num_anomaly < shot:
            print(
                f"[Warning] Class '{cls}': "
                f"using normal={num_normal}/{shot}, "
                f"anomaly={num_anomaly}/{shot}"
            )

        fewshot_data.extend(random.sample(normals, num_normal))
        fewshot_data.extend(random.sample(anomalies, num_anomaly))

    random.shuffle(fewshot_data)

    output_path = output_dir / f"{shot}-shot.jsonl"
    save_jsonl(fewshot_data, output_path)

    print(f"[✓] Saved {len(fewshot_data)} samples to {output_path}")
    print(f"    ({shot} normal + {shot} anomaly per class)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="Path to full-shot.jsonl")
    parser.add_argument(
        "--output_dir", required=True, help="e.g. ./dataset/metadata/VisA"
    )
    parser.add_argument(
        "--shot",
        type=int,
        required=True,
        help="x-shot = x normal + x anomaly per class",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
