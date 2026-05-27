# MEBeauty 多族裔臉部吸引力資料集與評分工具

本專案整理 MEBeauty 多族裔臉部吸引力資料集，並提供影像裁切對齊、PyTorch 訓練/推論、模型評估，以及本機 Web 版「高顏值照相機」。

資料集包含 2,550 張 Black、Asian、Caucasian、Hispanic、Indian、Mideastern 的女性與男性臉部圖片，由約 300 位不同文化與社會背景的評分者標註。平均分數與個人分數放在 `scores/`，常用訓練/測試檔包含 `scores/train_crop.csv`、`scores/test_crop.csv`、`scores/train_2022.txt`、`scores/test_2022.txt`、`scores/val_2022.txt`，完整分數表包含 `scores/generic_scores_all_2022.xlsx`、`scores/date_scores_all_2022.xlsx` 等。

資料蒐集、清理與分析流程在 `MEBeauty_creation_cleaning/`。原始圖片在 `original_images/`，不同方法裁切後的圖片在 `cropped_images/`。

![MEBeauty examples](ME3.png)

## 環境安裝

專案使用 `uv` 管理 Python 3.10 環境。DeepFace/TensorFlow 相關工具在 Windows 上以 Python 3.10 較穩定，因此 `pyproject.toml` 已限制 `>=3.10,<3.11`。

```powershell
uv sync
```

這會建立 `.venv`，並依照 `pyproject.toml` / `uv.lock` 安裝依賴。Windows 與 Linux 會使用 PyTorch CUDA 12.8 wheel；若沒有可用 GPU，程式會自動改用 CPU。

檢查 PyTorch 與 CUDA：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

更多 uv 使用方式請看 `UV_USAGE.md`。

## Web App：高顏值照相機

`web_app.py` 提供 FastAPI Web 介面，可上傳 JPG、PNG、WEBP、BMP，或使用瀏覽器相機拍照，並用 `pytorch_trained_models/` 內的本機 PyTorch 模型估計 1-10 分。

啟動：

```powershell
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
```

開啟：

```text
http://127.0.0.1:8000
```

目前 Web App 功能：

- 自動列出 `pytorch_trained_models/` 內的 `.pht`、`.pth`、`.pt` 模型。
- 使用 OpenCV 偵測最大臉部，對 tight、normal、wide 三種裁切結果取平均；偵測不到臉時使用整張圖片。
- 顯示分數、相對測試集百分位、各裁切分數與推論細節。
- 可設定顯示門檻值，分數低於門檻時會遮蔽臉部特徵；若 MTCNN 關鍵點可用，優先用關鍵點遮蔽，否則使用臉框橢圓遮蔽，沒有偵測到臉時會模糊整張圖。
- `best_model.pht` 的相對排名可直接使用 `pytorch_trained_models/prediction_inspection.csv` 快取。

## 臉部裁切與對齊

`face_crop_align.py` 會遞迴處理資料夾與子資料夾，保留原本目錄結構並輸出裁切後圖片。此功能基於 DeepFace。

```powershell
uv run python face_crop_align.py --images_path original_images --results_path crop_align_images --method opencv
```

參數：

- `--images_path`：來源圖片資料夾。
- `--results_path`：輸出資料夾；若資料夾已存在，程式會停止以避免覆蓋。
- `--method`：DeepFace 偵測後端，例如 `opencv`、`dlib`、`mtcnn`、`ssd`、`retinaface`。

預設後端是 `opencv`。

## PyTorch 訓練

資料集與 dataloader 在 `pytorch_mebeauty_dataset.py`。目前影像會 resize 到 `256x256`，並以 `[0.5, 0.5, 0.5]` 做 normalize。訓練時可啟用 RandomResizedCrop、水平翻轉、旋轉與 ColorJitter。

訓練指令：

```powershell
uv run python pytorch_train_val.py --base_model resnet18 --train_scores scores/train_crop.csv --test_scores scores/test_crop.csv --batch_size 32 --epochs 50 --train_augmentation true --lr 0.0001 --weight_decay 0.0001 --patience 6
```

支援的 base model：

- `resnet18`
- `resnet50`
- `efficientnet`
- `densenet`
- `mobilenet`
- `alexnet`
- `vgg16`

常用參數：

- `--loss`：`smoothl1` 或 `mse`，預設 `smoothl1`。
- `--freeze_backbone`：是否凍結 pretrained backbone，預設 `true`。
- `--dropout`：regression head dropout，預設 `0.35`。
- `--patience`：early stopping 等待 epoch 數，預設 `6`。
- `--num_workers`、`--pin_memory`：DataLoader 設定。

訓練輸出會寫到 `pytorch_trained_models/`：

```text
pytorch_trained_models/best_model.pht
pytorch_trained_models/last_model.pht
pytorch_trained_models/training_history.csv
```

## 單張圖片推論

```powershell
uv run python pytorch_predict.py --image_path inference_samples/girl.jpg --model_path pytorch_trained_models/best_model.pht
```

預設範例圖片是 `inference_samples/girl.jpg`，預設模型路徑是 `pytorch_trained_models/densenet_MSE_Adam_3_dropouts_nocrop.pht`。若該模型不存在，請改用 `pytorch_trained_models/best_model.pht` 或其他已訓練模型。

## 模型評估與檢查

評估 `pytorch_trained_models/` 內所有 `.pht` 模型，依 validation MSE 排名：

```powershell
uv run python evaluate_models.py --test_scores scores/test_crop.csv
```

輸出：

```text
pytorch_trained_models/model_ranking.csv
```

檢查單一模型在測試集上的預測分佈與誤差：

```powershell
uv run python inspect_predictions.py --model_path pytorch_trained_models/best_model.pht --scores scores/test_crop.csv
```

輸出：

```text
pytorch_trained_models/prediction_inspection.csv
```

## 其他特徵與 Notebook

傳統特徵與淺層模型：

- Eigenface、Gabor、HOG、Landmarks、SIFT 特徵萃取與淺層 predictor：`Eigenface,Geom.features,HOG,Gabor,SIFT+shallow predictor.ipynb`
- Landmark 與幾何特徵：`get_landmarks_geom.features.ipynb`
- 幾何特徵表：`geometric_features.csv`
- landmark 表：`landmarks.csv`

FaceNet 特徵：

- FaceNet 512 維 embedding 萃取與比較：`facenet 512 embedding extractors, facenet comparison.ipynb`
- 產出的特徵在 `FaceNet_512_features/`

啟動 Jupyter：

```powershell
uv run jupyter notebook
```

## 目錄速覽

```text
original_images/                 原始圖片
cropped_images/                  已裁切/對齊圖片
scores/                          平均分數、個人分數、train/test/val 清單
MEBeauty_creation_cleaning/      資料蒐集、清理與分析 notebook
FaceNet_512_features/            FaceNet embedding
inference_samples/               推論範例圖片
pytorch_trained_models/          訓練後模型與評估輸出
webapp/                          Web App 前端靜態檔
web_app.py                       FastAPI Web App
pytorch_train_val.py             PyTorch 訓練
pytorch_predict.py               單張圖片推論
evaluate_models.py               批次評估模型
inspect_predictions.py           檢查預測分佈與輸出明細
face_crop_align.py               臉部裁切與對齊
```

## 引用

如果你在研究中使用本資料集或程式碼，請引用：

```bibtex
@article{lebedeva2021mebeauty,
  title={MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author={Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2021},
  publisher={Springer}
}
```

## 使用限制與聯絡

MEBeauty 僅限非商業研究用途。

資料庫相關問題請聯絡原作者：irina.val.lebedeva@gmail.com
