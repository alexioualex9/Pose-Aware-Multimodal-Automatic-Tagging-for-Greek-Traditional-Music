# Pose-Aware Multimodal Automatic Tagging

This repository contains the code developed for the thesis project **“Pose-Aware Multimodal Automatic Tagging”**.

The project investigates whether **pose-derived motion information**, together with **audio** and **video**, can improve the **automatic tagging of Greek traditional music performances**. It explores how embodied performance cues such as **dance movement**, **posture**, and **visual context** can complement acoustic information in a culturally grounded music information retrieval setting.

The repository includes code for:

- **dance-scene detection**,
- **skeleton extraction from in-the-wild videos**,
- **unimodal processing pipelines** for audio, video, and skeleton modalities, and
- **multimodal fusion** across all combinations of audio, video, and skeleton representations.

The implemented framework supports automatic tagging on the **top-28 labels of the Lyra Dataset**, using both unimodal and multimodal models.

---

## Repository Structure

```text
.
├── detect_dance_scenes/
├── extract_skeletons/
├── skeletons/
├── video/
├── multimodal/
└── README.md
```

### Folder Description

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

## Dataset

All experiments in this thesis are based on the **Lyra Dataset**, a dataset of Greek traditional music performances annotated with multilabel semantic tags.

The full Lyra collection contains **1570 videos**. Among them, **767 videos include dancing** and form the initial **dance subset**. Since the pose-aware setting requires at least one valid extracted skeleton clip per video, the skeleton extraction and filtering stage reduces this subset to **749 videos**, referred to as the **skeleton subset**.

Following prior work on Lyra, this project begins from the **top-30 most frequent labels**. However, in the skeleton subset, two of these labels have zero positive support, so all experiments in this repository are conducted on the remaining **28 labels**.

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
