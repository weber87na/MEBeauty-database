import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pytorch_mebeauty_dataset import MEBeauty, build_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pytorch_trained_models/best_model.pht")
    parser.add_argument("--scores", default="scores/test_crop.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_csv", default="pytorch_trained_models/prediction_inspection.csv")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    dataset = MEBeauty("", args.scores, args.scores, train=False, transform=build_transform(False))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions = []
    targets = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            pred = model(xb).reshape(-1).detach().cpu()
            predictions.extend(pred.tolist())
            targets.extend(yb.float().tolist())

    df = pd.DataFrame({
        "image": dataset.images_scores.iloc[:, 0].tolist(),
        "target": targets,
        "prediction": predictions,
    })
    df["error"] = df["prediction"] - df["target"]

    print("device", device)
    print("target min/mean/max", df["target"].min(), df["target"].mean(), df["target"].max())
    print("pred   min/mean/max", df["prediction"].min(), df["prediction"].mean(), df["prediction"].max())
    print("pred quantiles")
    print(df["prediction"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print("count prediction >= 7:", int((df["prediction"] >= 7).sum()), "/", len(df))
    print("count target >= 7:", int((df["target"] >= 7).sum()), "/", len(df))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("prediction", ascending=False).to_csv(output_csv, index=False)
    print("saved", output_csv)


if __name__ == "__main__":
    main()
