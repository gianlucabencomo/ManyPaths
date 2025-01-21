# ManyPaths

This repository is a messy JAX / PyTorch implementation of the experiments for the Many Paths Project (README last updated: January 21st, 2025).

## Table of Contents

- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

---

## File Structure

Describe the contents and structure of the repository.

```
├── README.md
├── archive                 # Old scripts from previous project versions
├── bash                    # bash scripts to run on Della / Local
├── constants.py            # Constants to import for training / testing
├── datasets.py             # Pytorch Datasets
├── evaluation.py           # Meta-Testing scripts
├── figures                 # Output directory for figures
├── generate_concepts.py    # Script for generating images for concepts
├── generate_numbers.py     # Script for generating images for numbers
├── grammer.py              # Script for generating concepts
├── initialization.py       # Initialization Scripts
├── main.py                 # Meta-Training Scripts
├── models.py               # Pytorch Models for MLP, CNN, LSTM, and Transformer
├── omniglot                # Output directory for omniglot
├── requirements.txt        # Pipreqs
├── results                 # Output directory for saving results
├── slurm                   # Slurm scripts for Della
├── state_dicts             # Output directory for saving model states
├── summarize.py            # Scripts for printing Meta-Test results
├── test.py                 # Meta-testing script
├── training.py             # Meta-training / hyperparameter search script
├── utils.py                # Helper functions
└── visualize.py            # Visualization scripts
```

---

## Requirements

- Python 3.10
- Libraries specified in `requirements.txt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/ManyPaths.git
   cd ManyPaths
   ```

2. Create a virtual environment:
   ```bash
   # Using Python's venv
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate     # For Windows

   # Using Conda with Python 3.10
   conda create --name myenv python=3.10
   conda activate myenv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

The project provides CLI interfaces for both **meta-training** and **meta-testing** using `typer`. Below are detailed instructions for each:

### **Meta-Training** (`main.py`)
This script is used to train meta-learners using various architectures, datasets, and experiments.

#### **Basic Command**
```bash
python main.py
```

#### **CLI Options**

| Option                     | Default Value | Description                                                                                     |
|----------------------------|---------------|-------------------------------------------------------------------------------------------------|
| `--seed`                   | `0`           | Random seed for reproducibility.                                                               |
| `--experiment`             | `"mod"`       | Type of experiment to run. Options: `"mod"`, `"concept"`, `"omniglot"`.                        |
| `--m`                      | `"mlp"`       | Model architecture. Options: `"mlp"`, `"cnn"`, `"lstm"`, `"transformer"`.                      |
| `--data-type`              | `"image"`     | Type of data. Options: `"image"`, `"bits"`, `"number"`.                                        |
| `--a`                      | `"asian"`     | Alphabet type (used only for Omniglot). Options: `"ancient"`, `"asian"`, `"all"`.              |
| `--epochs`                 | `1000`        | Number of training epochs.                                                                     |
| `--tasks-per-meta-batch`   | `4`           | Number of tasks per meta-training batch.                                                       |
| `--adaptation-steps`       | `1`           | Number of adaptation steps for meta-learning.                                                  |
| `--outer-lr`               | `1e-3`        | Outer learning rate for meta-optimization.                                                     |
| `--skip`                   | `1`           | Data skipping parameter (affects dataset).                                                     |
| `--no-hyper-search`        | `False`       | Disable hyperparameter search. Set to `True` to skip hyperparameter optimization.              |
| `--plot`                   | `False`       | Enable plotting of training loss and evaluation results.                                       |
| `--save`                   | `False`       | Save the trained model after training.                                                         |

#### **Example Meta-Training Command**
```bash
python main.py --experiment mod --m cnn --plot
```

### **Meta-Testing** (`test.py`)
This script is used to evaluate pre-trained meta-learners on test datasets and generate results.

#### **Basic Command**
```bash
python test.py
```

#### **CLI Options**

| Option                 | Default Value       | Description                                                                                     |
|------------------------|---------------------|-------------------------------------------------------------------------------------------------|
| `--directory`          | `"./state_dicts/"` | Directory containing model state dictionaries (`.pth` files).                                   |
| `--output-folder`      | `"results"`        | Directory to save the results as CSV files.                                                    |
| `--experiment`         | `"mod"`           | Type of experiment to evaluate. Options: `"mod"`, `"concept"`, `"omniglot"`.                  |
| `--test-seeds`         | `10`               | Number of seeds to test during meta-testing.                                                   |
| `--adaptation-steps`   | `1`                | Number of adaptation steps for meta-testing.                                                   |
| `--plot`               | `False`            | Enable plotting of meta-test results.                                                          |
| `--save`               | `False`            | Save meta-testing results to the output folder.                                                |

#### **Example Meta-Testing Command**

1. **Run meta-testing with default settings:**
   ```bash
   python test.py --experiment mod --directory ./state_dicts/ --test-seeds 10
   ```

2. **Meta-test with custom output folder and enable plotting:**
   ```bash
   python test.py --experiment concept --output-folder test_results --plot
   ```

3. **Save results to a specific folder:**
   ```bash
   python test.py --experiment omniglot --save
   ```

---
