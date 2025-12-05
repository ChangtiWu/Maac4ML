# Maac4ML

The code of Maac4ML.

## Overview

This project studies how numerical precision control (decimal place rounding) and alternative activation functions affect model training and inference. Comparative experiments are conducted using ResNet18 and Vision Transformer architectures on the CIFAR-100 dataset.

## Project Structure

```
Maac4ML/
├── cnn/                          # CNN experiment module
│   ├── resnet_experiment.py      # ResNet18 training script
│   └── resnet_inference.py       # ResNet18 inference testing script
├── transformer/                  # Transformer experiment module
│   ├── vit_experiment.py         # Vision Transformer training script
│   └── vit_inference.py          # ViT inference testing script
├── utils/                        # Utility functions
│   └── plot_experiments.py       # Experiment results visualization tool
├── dataset/                      # Dataset storage
├── checkpoints/                  # Model checkpoints
├── log/                          # Training and inference logs
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- CUDA GPU (8GB+ VRAM recommended)

### Install Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Run ResNet18 Experiments

```bash
python cnn/resnet_experiment.py
```

Runs 3 experiments (20 epochs each):
- Experiment 1: Original precision + ReLU activation
- Experiment 2: 3 decimal places + Square activation
- Experiment 3: 6 decimal places + ReLU activation

### Run Vision Transformer Experiments

```bash
python transformer/vit_experiment.py
```

Runs 3 experiments (50 epochs each):
- Experiment 1: Original precision (no control)
- Experiment 2: 3 decimal places precision control
- Experiment 3: 6 decimal places precision control

### Inference Testing

```bash
# ResNet inference testing
python cnn/resnet_inference.py

# ViT inference testing
python transformer/vit_inference.py
```

### Visualize Results

```bash
# Generate ResNet experiment plots
python utils/plot_experiments.py -l log/resnet_experiment_YYYYMMDD_HHMMSS.jsonl -o resnet_results

# Generate ViT experiment plots
python utils/plot_experiments.py -l log/vit_experiment_YYYYMMDD_HHMMSS.jsonl -o vit_results
```

Output files:
- `*_train_loss.pdf` - Training loss comparison
- `*_train_acc.pdf` - Training accuracy comparison
- `*_test_acc.pdf` - Test accuracy comparison

## Hyper-parameters

### ResNet-18 Experiment Configuration

We use ResNet-18.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 20 | Number of training epochs |
| `batch_size` | 256 | Batch size |
| `learning_rate` | 0.01 | Initial learning rate |

### ViT-Small Experiment Configuration

We use Vit-Small, which uses CNN to do **Patch Embedding**, with 6 transformer blocks.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Number of training epochs |
| `batch_size` | 256 | Batch size |
| `learning_rate` | 0.001 with decay=0.05 | Initial learning rate with decay |
| `embed_dim` | 256 | Embedding dimension |
| `depth` | 6 | Number of transformer blocks |
| `num_heads` | 8 | Number of attention heads |

## Output Format

### Training Logs (JSONL)

```json
{
  "experiment_id": "exp1",
  "experiment_name": "Experiment 1: Original Precision + ReLU",
  "epoch": 1,
  "train_loss": 3.756,
  "train_acc": 12.43,
  "test_loss": 3.410,
  "test_acc": 17.35,
  "best_acc": 17.35,
  "epoch_time": 7.064
}
```

### Inference Results (JSON)

```json
{
  "test_name": "Test 1: Experiment 1 Model - Original Precision Inference",
  "training_last_acc": 67.56,
  "inference_acc": 67.56
}
```

## Notes

1. Ensure the dataset path is correctly configured
2. Modify GPU device numbers according to your setup
3. Ensure checkpoint directories exist and are writable
4. Scripts use random seed (42) for reproducibility

## License

MIT License
