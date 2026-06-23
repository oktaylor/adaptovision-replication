# AdaptoVision Replication on CIFAR-10

This repository contains an independent PyTorch replication of **AdaptoVision: A Multi-Resolution Image Recognition Model for Robust and Scalable Classification** on the CIFAR-10 dataset.

The goal of this project is to reproduce the CIFAR-10 image classification experiment from the paper as closely as possible, analyze the reproducibility of the reported result, and document the effect of implementation and training choices.

## Paper

* Paper: **AdaptoVision: A Multi-Resolution Image Recognition Model for Robust and Scalable Classification**
* arXiv: https://arxiv.org/abs/2504.12652
* Dataset used in this replication: **CIFAR-10**
* Training setting: **from scratch, no pretrained weights**
* Original reported CIFAR-10 accuracy: **95.3%**

## Repository Structure

```text
adaptovision-replication/
├── adaptovision/
│   ├── config.py
│   ├── dataset.py
│   ├── modeling/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── models/
│       ├── adaptovision.py
│       └── blocks.py
├── configs/
│   └── *.yaml
├── data/
│   └── raw/
├── scripts/
│   ├── create_env.sh
│   ├── train_cifar10.sh
│   ├── eval_cifar10.sh
│   └── analyze_runs.py
├── reports/
│   └── figures/
├── outputs/
│   ├── slurm/
│   ├── runs/
│   └── analysis/
├── requirements.txt
├── Makefile
├── pyproject.toml
└── README.md
```

The main implementation is in `adaptovision/`.

* `adaptovision/dataset.py`: CIFAR-10 dataset loading, train/validation/test split, and transforms
* `adaptovision/modeling/train.py`: training loop and checkpoint saving
* `adaptovision/modeling/evaluate.py`: test evaluation and confusion matrix generation
* `adaptovision/models/`: AdaptoVision model and block definitions
* `configs/`: experiment configuration files
* `scripts/`: environment setup, SLURM scripts, evaluation, and analysis scripts
* `outputs/runs/`: run-specific outputs, checkpoints, copied configs, metrics, and evaluation files
* `outputs/slurm/`: SLURM stdout/stderr logs
* `outputs/analysis/`: summarized experiment results and plots

Large files such as datasets, checkpoints, logs, and generated outputs are not intended to be tracked by Git.

## Environment Setup

This project uses Python 3.11 with a virtual environment.

To create the environment:

```bash
bash scripts/create_env.sh
```

Then activate the environment:

```bash
source .venv/bin/activate
```

The required Python packages are listed in:

```text
requirements.txt
```

## Dataset

This replication uses **CIFAR-10** only.

The dataset is downloaded automatically through `torchvision.datasets.CIFAR10` and stored under:

```text
data/raw/
```

The split used in this implementation is:

```text
45,000 training images
5,000 validation images
10,000 test images
```

The test set is used only for final evaluation. Checkpoints are selected based on validation accuracy.

## Configuration Files

Each experiment is controlled by a YAML configuration file in `configs/`.

A configuration file specifies the dataset, augmentation, model width, dropout schedule, optimizer, learning rate, batch size, scheduler, and run name.

Before submitting a training job, check the configuration file


## Running on the FIR Cluster

The experiments were run on the **Digital Research Alliance of Canada FIR cluster**.

The main SLURM training script is:

```text
scripts/train_cifar10.sh
```

This script:

1. loads the required cluster modules,
2. activates the existing virtual environment,
3. trains the CIFAR-10 model,
4. evaluates the best checkpoint on the test set.

Before submitting a job, open `scripts/train_cifar10.sh` and set:

```bash
CONFIG_PATH=configs/your_config_file.yaml
```

Then submit the job:

```bash
sbatch scripts/train_cifar10.sh
```

Training outputs are saved under:

```text
outputs/runs/
```

SLURM logs are saved under:

```text
outputs/slurm/
```

## Evaluation

The main training script already runs test evaluation after training.

To run evaluation separately with the default configuration:

```bash
sbatch scripts/eval_cifar10.sh
```

To evaluate using a specific configuration file:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/lr0030_mild_aug_drop_mid_800ep_bs256.yaml scripts/eval_cifar10.sh
```

Evaluation can also be run manually:

```bash
source .venv/bin/activate

python -m adaptovision.modeling.evaluate \
  --config configs/lr0030_mild_aug_drop_mid_800ep_bs256.yaml
```

If a checkpoint path is not provided, the evaluation script uses the run information in the config file to locate the corresponding checkpoint.

## Analyzing Experiment Runs

To summarize completed runs:

```bash
python scripts/analyze_runs.py \
  --runs-dir outputs/runs \
  --out outputs/analysis \
  --top-k 8
```

This script summarizes training metrics and evaluation results from `outputs/runs/` and saves output files to:

```text
outputs/analysis/
```

Typical analysis outputs include:

```text
summary.csv
summary.md
test_accuracy_by_run.png
lr_vs_test_accuracy.png
top_val_accuracy_curves.png
top_train_accuracy_curves.png
top_val_loss_curves.png
best_confusion_matrix_normalized.png
best_per_class_recall.png
best_top_confusions.csv
```

## Output Structure

A typical run directory is saved as:

```text
outputs/runs/<TIMESTAMP>_CIFAR10_<RUN_NAME>/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── config.yaml
├── metrics.csv
├── train_summary.txt
└── evaluation/
    ├── test_results.txt
    └── confusion_matrix.txt
```

The main checkpoint used for evaluation is:

```text
checkpoints/best.pt
```

This checkpoint is selected based on the best validation accuracy.

## Main Results

The original paper reported **95.3%** accuracy on CIFAR-10.

In this independent replication, the strict reported setting with the original learning rate was not stable in our implementation. High-learning-rate warmup experiments were also tested, but they did not match the lower-learning-rate baselines.

| Experiment                       | Test Accuracy |
| -------------------------------- | ------------: |
| Original AdaptoVision paper      |        95.30% |
| Strict reported setting          |  Failed / NaN |
| Reported LR recovery with warmup |        80.57% |
| Stable LR modification           |        83.44% |
| Dropout modification             |        84.17% |
| Augmentation modification        |        85.33% |
| Best combined ablation           |        86.04% |

The best overall result was obtained with a modified setting using milder geometric augmentation, reduced dropout, and batch size 256.

The best result remains below the paper-reported accuracy. The final report discusses likely reasons, including architectural ambiguity, missing implementation details, learning-rate sensitivity, augmentation strength, and regularization effects.

## Notes

This repository only includes the CIFAR-10 replication experiment.

No pretrained AdaptoVision checkpoint or official AdaptoVision implementation was used. The model was implemented independently based on the paper description.

The paper description contains some architectural ambiguity, especially in the relationship between the equations and figures for Block-1 and Block-2. This implementation follows the equation-based lightweight interpretation where necessary to keep the model close to the reported parameter scale.
