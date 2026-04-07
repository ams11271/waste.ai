# waste.ai — Deep Learning Waste Classification

An end-to-end computer vision system that classifies waste materials into **6 recyclable categories** using transfer learning with PyTorch. Trained on ~2,500 labeled images, the model enables automated sorting for smarter recycling and waste management.

---

## The Problem

Contaminated recycling streams are one of the biggest obstacles in waste management — a single misclassified item can ruin an entire batch. Manual sorting is slow, expensive, and error-prone. **waste.ai** automates this process using deep learning to classify waste from images in real time.

---

## Results

| Metric | Value |
|---|---|
| Waste categories classified | 6 (cardboard, glass, metal, paper, plastic, trash) |
| Training images | ~2,500 across all classes |
| Model architecture | Pre-trained CNN with transfer learning (PyTorch) |
| Saved model size | 94.4 MB (`trained_model1.pth`) |

---

## Classification Categories

| Category | Sample Count | Description |
|---|---|---|
| Paper | 594 | Newspapers, office paper, magazines |
| Glass | 501 | Bottles, jars, broken glass |
| Plastic | 482 | Bottles, containers, packaging |
| Metal | 410 | Cans, foil, scrap metal |
| Cardboard | 403 | Boxes, packaging, corrugated material |
| Trash | 137 | Non-recyclable general waste |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3** | Core language |
| **PyTorch** | Deep learning framework — model training, inference, and optimization |
| **Torchvision** | Pre-trained CNN architectures and image transformation pipeline |
| **Matplotlib** | Data visualization and sample inspection |
| **Jupyter Notebook** | Interactive experimentation and iterative model development |

---

## How It Was Built

### 1. Dataset Preparation
- Sourced a labeled dataset of **2,527 waste images** organized into 6 category folders
- Created train/test/validation splits (with and without the trash category) for rigorous evaluation
- Built indexed mapping files for reproducible data loading

### 2. Image Preprocessing Pipeline
- Resized all images to uniform dimensions (256x256 / 128x128) for batch processing
- Applied `transforms.Compose()` pipeline: resize, tensor conversion, and normalization
- Used `ImageFolder` for structured dataset loading directly from the directory hierarchy

### 3. Model Training — Transfer Learning
- Leveraged **pre-trained convolutional neural networks** from Torchvision as feature extractors
- Fine-tuned the classification head on the waste dataset — achieving strong accuracy with limited training data
- Experimented across multiple notebook iterations to optimize architecture and hyperparameters
- Exported the final trained model as `trained_model1.pth` for deployment

### 4. Inference & Visualization
- Built sample visualization utilities to inspect predictions against ground truth
- Tested the model on unseen example images (bottles, cans, mixed waste) to validate generalization

---

## Project Structure

```
waste.ai/
├── garbage_classification.ipynb    # Primary training notebook (256x256 images)
├── Pytorch.ipynb                   # Transfer learning experiments
├── Pytorch copy 2.ipynb            # Architecture iteration (128x128 images)
├── trained_model1.pth              # Saved PyTorch model weights
├── archive (1)/                    # Full labeled dataset
│   └── Garbage classification/
│       ├── cardboard/              # 403 images
│       ├── glass/                  # 501 images
│       ├── metal/                  # 410 images
│       ├── paper/                  # 594 images
│       ├── plastic/                # 482 images
│       └── trash/                  # 137 images
├── *.jpg / *.webp                  # Test images for inference
└── README.md
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ams11271/waste.ai.git
cd waste.ai

# Install dependencies
pip install torch torchvision matplotlib jupyter

# Launch the training notebook
jupyter notebook garbage_classification.ipynb
```

To run inference with the pre-trained model:
```python
import torch
model = torch.load('trained_model1.pth', map_location='cpu')
model.eval()
```

---

## Real-World Applications

- **Smart recycling bins** — automatically sort waste at the point of disposal
- **Municipal waste processing** — reduce contamination in recycling streams
- **Environmental monitoring** — classify litter in public spaces from camera feeds
- **Education** — teach communities proper waste sorting through instant feedback

---

## About

Built to explore practical applications of computer vision in sustainability. This project demonstrates end-to-end ML skills: dataset curation, image preprocessing, transfer learning, model iteration, and deployment-ready artifact generation.
