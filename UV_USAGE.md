# UV usage

This project uses `uv` with Python 3.10 because the DeepFace/TensorFlow stack used by `face_crop_align.py` is most stable on that version for Windows.

## First setup

From the project root:

```powershell
uv sync
```

This creates `.venv` and installs the dependencies from `pyproject.toml` / `uv.lock`.

On Windows and Linux, PyTorch is configured to use the CUDA 12.8 wheels from the official PyTorch index. Check GPU availability with:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Run scripts

Use `uv run` so commands execute inside the project environment:

```powershell
uv run python face_crop_align.py --images_path original_images --results_path crop_align_images --method opencv
```

```powershell
uv run python pytorch_predict.py --image_path inference_samples/girl.jpg --model_path pytorch_trained_models/densenet_MSE_Adam_3_dropouts_nocrop.pht
```

```powershell
uv run python pytorch_train_val.py --base_model vgg16 --train_scores scores/train_crop.csv --test_scores scores/test_crop.csv --batch_size 16 --epochs 25
```

Recommended training command for a smaller dataset:

```powershell
uv run python pytorch_train_val.py --base_model resnet18 --train_scores scores/train_crop.csv --test_scores scores/test_crop.csv --batch_size 32 --epochs 50 --train_augmentation true --lr 0.0001 --weight_decay 0.0001 --patience 6
```

The optimized training script writes:

```text
pytorch_trained_models/best_model.pht
pytorch_trained_models/last_model.pht
pytorch_trained_models/training_history.csv
```

Rank existing model files by validation performance:

```powershell
uv run python evaluate_models.py --test_scores scores/test_crop.csv
```

## Activate the virtual environment manually

Activation is optional because `uv run` is usually enough.

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, regular `python` and `pip` commands use `.venv`.

## Jupyter notebooks

Start Jupyter from the project environment:

```powershell
uv run jupyter notebook
```

If you need a named kernel:

```powershell
uv run python -m ipykernel install --user --name mebeauty --display-name "Python (MEBeauty)"
```

## Dependency changes

Add a package:

```powershell
uv add package-name
```

Remove a package:

```powershell
uv remove package-name
```

Reinstall from the lockfile:

```powershell
uv sync --locked
```
