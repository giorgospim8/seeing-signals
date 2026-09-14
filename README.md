# Seeing Signals: Automatic Modulation Classification

Automatic modulation classification from constellation diagrams using deep learning and vision-language models.

This academic project was developed for the **Communication Systems II** course at the **Aristotle University of Thessaloniki (AUTH)**. It investigates whether communication signals can be automatically characterized directly from constellation diagram images under different channel and hardware impairments.

## Project Overview

The project follows a three-stage pipeline:

1. **Signal Modeling & Dataset Generation**  
   Generation of IQ samples and constellation diagrams across 16 digital modulation schemes, with simulated channel and hardware impairments including AWGN, phase noise, I/Q imbalance, and jamming.

2. **Multi-Task CNN Classification**  
   Development of a custom convolutional neural network with shared feature extraction and five classification heads for:
   - Modulation scheme
   - SNR range
   - Phase-noise severity
   - I/Q imbalance severity
   - Jamming detection

3. **Vision-Language Model Fine-Tuning**  
   Fine-tuning of **SmolVLM-256M-Instruct** using **LoRA** and comparison of its classification performance with the CNN baseline.

## Dataset

The generated dataset contains:

- **16 modulation schemes**
- **30,720 constellation images**
- **224 × 224 pixel images**
- Multiple levels of SNR, phase noise, I/Q imbalance, and jamming
- Structured labels stored in CSV and JSON formats

The data is split into **70% training, 15% validation, and 15% testing**.

The full generated dataset is not included in this repository. It can be reproduced using the dataset-generation notebook.

## Results

The multi-task CNN achieved the following test accuracies:

| Task | Accuracy |
| --- | ---: |
| Modulation classification | 58.36% |
| SNR classification | 85.03% |
| Phase-noise classification | 43.19% |
| I/Q imbalance classification | 40.54% |
| Jamming detection | 84.35% |

The CNN substantially outperformed the fine-tuned VLM on the evaluated classification tasks, highlighting the effectiveness of task-specific convolutional architectures for constellation-diagram analysis.

## Repository Structure

```text
seeing-signals/
├── notebooks/
│   ├── 01_signal_modeling_and_dataset_generation.ipynb
│   ├── 02_cnn_multitask_classification.ipynb
│   └── 03_vlm_lora_finetuning.ipynb
└── .gitignore
```

## Technologies

**Python · PyTorch · NumPy · Pandas · scikit-learn · Hugging Face Transformers · PEFT / LoRA · Matplotlib**

## Authors

Academic team project by **Georgios Poimenidis** and **Charalampos Ioannidis**  
Aristotle University of Thessaloniki — School of Electrical and Computer Engineering
