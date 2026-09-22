# Pose-Aware Multimodal Automatic Tagging

This repository contains the code developed for the paper **“Pose-Aware Multimodal Automatic Tagging”**.

The project investigates whether **pose-derived motion information**, together with **audio** and **video**, can improve the **automatic tagging**. It explores how embodied performance cues such as **dance movement**, **posture**, and **visual context** can complement acoustic information in a culturally grounded music information retrieval setting.

The implemented framework supports automatic tagging on the **top-28 labels of the Lyra Dataset**, using both unimodal and multimodal models.

---

## Repository Structure

- **detect_dance_scenes/**  
  Training and inference code for identifying dance-related scenes in video recordings.

- **extract_skeletons/**  
  Utilities for pose estimation, person tracking, primary-dancer selection, and skeleton extraction from video material.

- **skeletons/**  
  Skeleton-based unimodal processing, embedding generation, training, and evaluation.

- **video/**  
  Video embedding extraction, unimodal video training, evaluation, and late-fusion utilities.

- **multimodal/**  
  Architectures and training scripts for multimodal fusion across audio, video, and skeleton representations.

---

## Installation

The project was developed using **Python 3.8.20** and **PyTorch 2.2.0** with **CUDA 11.8**.

Install the dependencies required for training and evaluation:

```bash
python -m pip install -r requirements.txt
```

For dance-scene detection and skeleton extraction, install the additional dependencies:

```bash
python -m pip install -r requirements_pose.txt
```

The pose-extraction pipeline uses the following pretrained configurations and checkpoints:

- AlphaPose config: `configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml`
- AlphaPose checkpoint: `pretrained_models/fast_res50_256x192.pth`
- ByteTrack config: `exps/example/mot/yolox_x_mix_det.py`
- ByteTrack checkpoint: `pretrained/bytetrack_x_mot17.pth.tar`

AlphaPose and ByteTrack are installed from the exact Git commits specified in `requirements_pose.txt`. Their pretrained checkpoints are not included in this repository and must be downloaded from their official repositories. Local paths to datasets, external repositories, and checkpoints must be specified in the corresponding configuration files.

## Dataset

All experiments in this paper are based on the **Lyra Dataset**, a dataset of Greek traditional music performances annotated with multilabel semantic tags.

The full Lyra collection contains **1570 videos**. Among them, **767 videos include dancing**. Since the pose-aware setting requires at least one valid extracted skeleton clip per video, the skeleton extraction and filtering stage reduces this subset to **749 videos**, referred to as the **pose subset of Lyra dataset**.

Following prior work on Lyra, this project begins from the **top-30 most frequent labels**. However, in the pose subset of Lyra dataset, two of these labels have zero positive support, so all experiments in this repository are conducted on the remaining **28 labels**.

Depending on the experiment, the code may require:

- video recordings,
- annotation or label files,
- pretrained checkpoints,
- metadata associated with the Lyra Dataset.

This repository does **not necessarily include the raw datasets** used in the experiments. Users are expected to provide their own local data paths and organize the required files according to the input requirements of each script.

---

## Usage

### Train the Dance-Scene Detector

The following script fine-tunes the dance-scene detection model. The detector is trained on **1-second clips**.

```bash
python detect_dance_scenes/train_dance_detector.py \
  --video-dir <VIDEO_DIR> \
  --labels-file <LABELS_FILE> \
  --output-dir <OUTPUT_DIR> \
  --epochs <NUM_EPOCHS> \
  --mode full
```

### Apply Dance-Scene Detection

The following script applies a trained detector to identify dance-related scenes in video recordings.

```bash
python -m detect_dance_scenes/main.py \
  --video-dir <VIDEO_DIR> \
  --model-path <MODEL_PATH> \
  --output-dir <OUTPUT_DIR>
```

This stage processes the input videos, detects scene boundaries, performs clip-level inference, and stores the detected dance-scene intervals in output files.

---

### Skeleton-Based Processing

This stage includes:

1. trimming videos to their detected dance scenes,
2. applying **ByteTrack** for multi-person tracking,
3. selecting the **primary dancer**,
4. applying **AlphaPose** for pose estimation, and
5. storing keypoints together with metadata in `.json` format.

```bash
python -m extract_skeletons/main.py
```

---

### Create Skeleton Embeddings

This step creates skeleton embeddings for the training, validation, and test splits after selecting **T = 32 skeletons** from each clip.

The selection process is based on a quality-aware pipeline including:

- normalization,
- interpolation,
- joint confidence,
- bone-length consistency,
- left-right symmetry,
- temporal jitter penalties,
- skeleton similarity.

```bash
python -m skeletons/main.py process_skeleton_sequences --set train --device cuda
python -m skeletons/main.py process_skeleton_sequences --set val --device cuda
python -m skeletons/main.py process_skeleton_sequences --set test --device cuda
```

---

### Train the STGCN-like Skeleton Model

The skeleton-based model used in this work is a lightweight GCN architecture that includes:

- ST-GCN-like blocks with **64 channels**,
- multi-scale temporal convolutions with kernel sizes **9** and **3**,
- residual connections,
- global average pooling,
- a linear multilabel classification head.

In contrast to deeper adaptive variants, the graph adjacency remains fixed throughout training, which keeps the model lightweight and helps reduce overfitting when noisy skeleton inputs are used.

```bash
python -m skeletons/main.py train --model_name STGCN --device cuda
```

### Evaluate the STGCN-like Skeleton Model

```bash
python -m skeletons/main.py eval --model_name STGCN --device cuda
```

---

### Extract Video Embeddings

In this setup, video embeddings are extracted using one of five pretrained video models.

```bash
python -m video/extract_video_embeddings/extract_embeddings.py --dataset "lyra" --audio_model_name "ast" --seed {42, 123, 1337, 2024, 9999} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

---

### Train the Video Model

In this setup, the video model is trained using frozen embeddings extracted in the previous step.

```bash
python -m video/train.py --dataset "lyra" --time_window "8.00" --subset {"True", "False"} --embs "frozen" --seed {42, 123, 1337, 2024, 9999} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

---

### Evaluate the Video Model

```bash
python -m video/eval.py --dataset "lyra" --time_window "8.00" --subset {"True", "False"} --embs "frozen" --seed {int} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

---

### Late Fusion

Late fusion is applied by aggregating the output probabilities of each modality after extracting modality-specific predictions.

```bash
python -m video/late_fusion.py --modalities {'a,v', 'a,s', 'v,s', 'a,v,s'} --fusion {"weighted", "mean", "sum"} --weights {"equal", "f1_macro"} --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --video_model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --skeleton_model_name "STGCN"
```

---

### Train a Multimodal Model

The repository supports multiple multimodal fusion settings. In the current implementation:

- **simple transformer** performs early fusion,
- **gated fusion** learns clip-level modality weighting,
- **cross-attention fusion** is inspired by MulT-style cross-modal interaction.

```bash
python -m mutimodal/transformer.py --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --standardize --model_name {"seq_transformer_avs_masked", "seq_transformer_as_masked", "seq_transformer_vs_masked", "seq_transformer_av"} --transformer {"simple, "gated", "cros_attention"} --device {"cpu", "cuda"}
```

### Evaluate a Multimodal Model

```bash
python -m mutimodal/transformer.py --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --standardize --model_name {"seq_transformer_avs_masked", "seq_transformer_as_masked", "seq_transformer_vs_masked", "seq_transformer_av"} --transformer {"simple, "gated", "cros_attention"} --device {"cpu", "cuda"} --eval_only
```

---

## Citation

If you use this repository in academic work, please cite the corresponding thesis and/or paper once available.

---

## Author

**Alexandros Alexiou**
