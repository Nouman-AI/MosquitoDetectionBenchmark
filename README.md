# Mosquito Detection Benchmark

**Benchmarking deep-learning and classical computer-vision methods for mosquito detection and localization**

This repository provides Python code, detector configurations, and trained-model resources associated with a comparison of 23 methods for detecting *Aedes aegypti* mosquitoes in laboratory images. The benchmark examines detector choice across bounding-box accuracy, operating-point detection, centroid localization, known-count localization, and computational performance.

## Methods

The study compares four YOLO detectors, six TorchVision detectors, and thirteen classical computer-vision methods.

| Family | Methods |
|---|---|
| YOLO (4) | YOLO11n, YOLO11s, YOLOv8n, YOLOv8s |
| TorchVision (6) | Faster R-CNN ResNet50-FPN-v2, Faster R-CNN MobileNetV3-Large-FPN, Mask R-CNN ResNet50-FPN-v2, RetinaNet ResNet50-FPN-v2, SSD300-VGG16, SSDLite320-MobileNetV3-Large |
| Classical (13) | Difference of Gaussians, Laplacian of Gaussian, MSER, Otsu thresholding, adaptive Gaussian thresholding, black-hat morphology, local minima, Canny contours, median-background differencing, MOG2, KNN, frame differencing, SimpleBlob |

Method identifiers used in the code and the selected configuration files correspond to the methods in the manuscript.

## Evaluation framework

The benchmark distinguishes three evaluation settings:

1. **Ranked detection:** AP at IoU = 0.50 and mean AP across IoU thresholds 0.50–0.95, without supplying the number of mosquitoes as a prediction constraint.
2. **Unknown-count operating point:** precision, recall, and F1 computed from the detector outputs using validation-selected operating settings.
3. **Known-count localization:** an additional analysis using the experimental count of five mosquitoes per image, evaluated separately from unconstrained detection.

Centroid localization is assessed across distance tolerances, and CPU/GPU inference throughput is reported separately. The supplementary material contains the reported performance tables and figures.

## Data and annotation protocol

The manuscript describes 20 independent ten-minute video recordings, separated by recording into 14 training, three validation, and three test recordings. The described complete partitions contain 7,000 training images with 35,000 boxes, 1,500 validation images with 7,500 boxes, and 1,500 test images with 7,500 boxes. Each selected image contains five annotated mosquitoes. Bounding boxes were manually prepared in Make Sense and stored in Pascal VOC XML format.

The original development experiment used a smaller sampled training/validation subset; its selected checkpoints and parameters were retained for the subsequent evaluation on the expanded 1,500-image test set. The reported experimental results therefore do not represent fresh retraining on all 7,000 training images.

The data supporting the findings of this study may be made available by the corresponding author upon reasonable request, subject to applicable institutional approvals and data-sharing restrictions.

## Repository files

| File or directory | Contents |
|---|---|
| `train.py` | YOLO and TorchVision training entry point |
| `inference.py` | Prediction generation for the 23 supported methods |
| `evaluate.py` | Ranked AP, operating-point detection, known-five, and centroid evaluation |
| `benchmark_utils.py` | Dataset loading, model architectures, checkpoint loading, and common utilities |
| `classical.py` | Classical detector implementations |
| `hyperparameters/` | Candidate hyperparameter grids (JSON) and selected configurations (CSV) |
| `model_weights/` | Trained deep-learning checkpoint files and checkpoint documentation |
| `requirements.txt` | Python package requirements |

## Software requirements

Python, PyTorch, TorchVision, Ultralytics, OpenCV, NumPy, Pillow, and PyYAML are used in the provided scripts. The package requirements are listed in `requirements.txt`. Compatible PyTorch and TorchVision builds are required for the selected CPU or CUDA environment.

```bash
pip install -r requirements.txt
```

The requirements specify supported package ranges for this portable code release rather than a frozen record of the original training environment.

## Data layout for running the scripts

The dataset is not included in this repository. The scripts accept Pascal VOC image and annotation pairs organized using the following directory structure:

```text
data/
  train/
    images/
    annotations/
  val/
    images/
    annotations/
  test/
    images/
    annotations/
```

Image and annotation files should share the same base name. Image folders may contain nested subdirectories, with corresponding XML files in the annotation folders.

## Training

The training entry point supports the four YOLO and six TorchVision architectures.

```bash
python train.py --data data --method Y03_yolov8n
python train.py --data data --method T01_frcnn_r50
```

For YOLO, training uses the configuration recorded in `hyperparameters/selected/YOLO_training_settings.csv`. TorchVision training configurations are read from the supplied JSON files. Model selection is based on validation mAP50–95. The Mask R-CNN implementation derives rectangular training masks from VOC bounding boxes rather than externally annotated instance masks.

The scripts provide a portable training implementation. The historical experimental checkpoints were retained from an earlier development run; retraining may produce different checkpoints and metrics.

## Inference

The inference entry point supports individual detector IDs or all methods. Trained deep-learning checkpoints are read from `model_weights/`; classical configurations are read from the selected parameter records.

```bash
python inference.py --data data --split test --method Y03_yolov8n
python inference.py --data data --split test --method T01_frcnn_r50
python inference.py --data data --split test --method N01_dog
python inference.py --data data --split test --method all
```

Predictions are written as per-method JSON files under `predictions/`. They contain image identifiers, `xyxy` bounding boxes, and detector scores. The confidence scores of different detector families are not calibrated to a common scale.

## Evaluation

```bash
python evaluate.py --data data --split test --predictions predictions
```

The evaluation writes `summary.csv`, `ap_by_iou.csv`, and `centroid_metrics.csv` to `evaluation_results/`. The portable implementation uses confidence-ranked matching, interpolated AP, IoU-based operating metrics, a separate five-detection analysis, and centroid distance matching. Its inference and metric results depend on the selected checkpoint, original annotation version, and evaluation settings.

## Reproducibility and scope

The public scripts implement the principal model families and evaluation tasks using the parameter records distributed with the repository. They are **portable reference implementations**, not a byte-for-byte archive of the historical training and test executables. In particular, the final reported test results were produced using previously retained model checkpoints and validation selections, together with the experimental evaluation workflow. Consequently, running the portable scripts does not independently guarantee reproduction of every numerical value in the manuscript.

The selected parameter files record the final configurations, whereas the JSON grids record candidate configurations. Numerical benchmark results and detailed method comparisons are reported in the manuscript and accompanying supplementary material. The original recordings and annotation data are not included here.

## Trained Model Weights

The trained checkpoints for all ten deep-learning models evaluated in the benchmark are publicly available through Zenodo.

Zenodo: https://doi.org/10.5281/zenodo.24021658

The archive contains four YOLO models and six TorchVision models. The checkpoint filenames identify the corresponding model architectures and method identifiers.

Citation:

Javed, N. (2026). Trained Deep Learning Models for Mosquito Detection Benchmarking (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24021658

## Citation

The manuscript citation will be added upon publication. The repository is intended to accompany the benchmark paper and its supplementary material.
