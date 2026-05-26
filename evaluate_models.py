import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_mebeauty_dataset import MEBeauty, build_transform


def evaluate_model(model_path, dataloader, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")
    mse_sum = 0.0
    mae_sum = 0.0
    total = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            preds = model(xb).reshape(-1)
            mse_sum += mse_loss(preds, yb).item()
            mae_sum += mae_loss(preds, yb).item()
            total += len(yb)

    return {
        "mse": mse_sum / total,
        "mae": mae_sum / total,
        "samples": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="pytorch_trained_models")
    parser.add_argument("--test_scores", default="scores/test_crop.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_csv", default="pytorch_trained_models/model_ranking.csv")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Evaluating on", device)

    model_dir = Path(args.model_dir)
    model_paths = sorted(model_dir.glob("*.pht"))
    if not model_paths:
        raise FileNotFoundError(f"No .pht models found in {model_dir}")

    transform_test = build_transform(False)
    dataset = MEBeauty("", args.test_scores, args.test_scores, train=False, transform=transform_test)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    for index, model_path in enumerate(model_paths, start=1):
        print(f"[{index}/{len(model_paths)}] {model_path.name}")
        metrics = evaluate_model(model_path, dataloader, device)
        rows.append({
            "model": str(model_path),
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "samples": metrics["samples"],
        })
        print("  mse={:.4f} mae={:.4f}".format(metrics["mse"], metrics["mae"]))

    ranking = pd.DataFrame(rows).sort_values("mse", ascending=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_csv, index=False)

    best = ranking.iloc[0]
    print()
    print("Best model by validation MSE:")
    print(best["model"])
    print("mse={:.4f} mae={:.4f}".format(best["mse"], best["mae"]))
    print("Saved ranking to", output_csv)


if __name__ == "__main__":
    main()
