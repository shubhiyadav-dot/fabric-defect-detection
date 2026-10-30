import os
import csv
import base64
import time
import argparse
from pathlib import Path

import requests

API_URL = (
    "https://serverless.roboflow.com/"
    "sanya-mohanty/workflows/"
    "fabric-defect-detection-idm-2-vfabric-defect-detection-idm-2-2-resnet18-t1-logic"
)

def predict_image(image_path, api_key):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    start_time = time.perf_counter()

    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": encoded_image
                }
            }
        },
        timeout=120
    )

    latency_ms = (time.perf_counter() - start_time) * 1000
    response.raise_for_status()

    result = response.json()
    outputs = result.get("outputs", [])

    if not outputs:
        raise ValueError("No outputs returned by Roboflow")

    prediction_data = outputs[0].get("predictions", {})
    predicted_classes = prediction_data.get("predicted_classes", [])

    if predicted_classes:
        predicted_label = predicted_classes[0]
    else:
        scores = prediction_data.get("predictions", {})
        if not scores:
            raise ValueError("No class predictions returned")
        predicted_label = max(
            scores,
            key=lambda name: scores[name]["confidence"]
        )

    confidence = prediction_data.get(
        "predictions", {}
    ).get(predicted_label, {}).get("confidence", 0)

    return predicted_label, confidence, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing test images"
    )
    parser.add_argument(
        "--output",
        default="predictions.csv"
    )
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "Set ROBOFLOW_API_KEY environment variable first."
        )

    image_folder = Path(args.input)

    if not image_folder.is_dir():
        raise FileNotFoundError(image_folder)

    image_files = sorted(
        p for p in image_folder.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_files:
        raise ValueError("No supported images found")

    results = []

    for index, image_path in enumerate(image_files, 1):
        try:
            label, confidence, latency = predict_image(
                image_path, api_key
            )

            results.append({
                "image": image_path.name,
                "predicted_label": label,
                "confidence": confidence,
                "latency_ms": round(latency, 2),
                "status": "success"
            })

            print(f"{index}/{len(image_files)}: {label}")

        except Exception as error:
            results.append({
                "image": image_path.name,
                "predicted_label": "",
                "confidence": "",
                "latency_ms": "",
                "status": f"error: {type(error)._name_}"
            })
            print(f"{index}/{len(image_files)}: failed")

    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "predicted_label",
                "confidence",
                "latency_ms",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Completed: {len(results)} images")
    print(f"Results saved to {args.output}")


if _name_ == "_main_":
    main()
