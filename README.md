# PAC Fault Diagnosis with Knowledge Graph + Bayesian Neural Network

A machine learning project for Packaged Air Conditioner (PAC) fault diagnosis using Knowledge Graph features and Bayesian Neural Networks with evidence framework.


## Project Structure

```
PAC-Fault-Diagnosis-KG-BNN/
├── config.py              # Central configuration
├── preprocessing.py       # Data loading, MinMax scaling & stratified splits
├── model.py               # BNN with evidence framework
├── baselines.py           # baseline classifiers
├── train.py               # Training loop (Adam, GPU)
├── evaluate.py            # Precision / Recall / F1 metrics
├── main.py                # 5-fold runs → Table 2 output
└── data/                  # Dataset directory (not in repo)
    ├── features.csv       # features 
    └── labels.csv         # Class labels (0-3)
```

## Usage

### 1. Prepare Your Data

Place your dataset in the `data/` directory:
- `features.csv`
- `labels.csv`: Single column "label" with integer class IDs (0-3)

### 2. Run Experiments

```bash
python main.py
```

This will:
- Load the dataset from `data/`
- Run 5-fold cross-validation
- Train BNN + 6 baselines on each fold
- Print Table 2 with Mean ± Std of Precision, Recall, F1-score
