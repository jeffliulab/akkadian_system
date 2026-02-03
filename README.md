# Akkadian Translation System

A deep learning-based translation tool for ancient cuneiform script.

## Key Features

### MLOps Workflow

- **End-to-End Deployment**: FastAPI backend + frontend interface with automatic GitHub Pages deployment
- **HPC Training Support**: Integrated SLURM scheduling scripts for high-performance computing cluster training
- **Model Version Management**: Modularized preprocessing, training, and inference pipeline

### Model Architectures

#### Naive Transformer

- Transformer architecture implemented from scratch
- Custom optimizer and training pipeline
- Complete checkpoint management with incremental training support

#### ByT5-Small Fine-tuning

- Fine-tuned from Google's ByT5 pretrained model (byte-level T5)
- Specialized for Akkadian-to-English translation tasks
- Complete data preprocessing and inference pipeline

## Project Structure

```
├── backend/          # FastAPI server
├── frontend/         # Web interface
└── notebooks/        # Training scripts
    ├── naive_transformer/   # Custom Transformer
    ├── byt5_small/          # ByT5 fine-tuning
    └── byt5_base/           # ByT5-base training
```

.
