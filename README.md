# Facial Emotion Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A CNN-based deep learning project for recognizing human emotions from facial images. The model classifies faces into seven emotion categories: **angry**, **disgust**, **fear**, **happy**, **sad**, **surprise**, and **neutral**.

## 🎯 Objectives

1. Implement a baseline CNN for facial emotion classification
2. Apply transfer learning using pretrained architectures (ResNet18/VGG16)
3. Compare feature extraction vs. fine-tuning approaches
4. Visualize learned representations using Grad-CAM

## 📊 Dataset

This project uses the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle:

| Split | Images |
|-------|--------|
| Train | 28,709 |
| Validation | 3,589 |
| Test | 3,589 |
| **Total** | **35,887** |

- Image size: 48×48 grayscale
- 7 emotion categories

## 🏗️ Project Structure

```
facial-emotion-recognition/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   └── config.yaml           # Training configurations
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py   # LeNet-inspired baseline model
│   │   └── transfer_model.py # ResNet18/VGG16 transfer learning
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset loading and preprocessing
│   │   └── augmentation.py   # Data augmentation transforms
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gradcam.py        # Grad-CAM visualization
│   │   ├── visualization.py  # Plotting utilities
│   │   └── metrics.py        # Evaluation metrics
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_transfer_learning.ipynb
├── tests/
│   └── test_models.py
├── checkpoints/              # Saved model weights
├── results/                  # Training logs and visualizations
└── data/                     # Dataset directory (not tracked)
    └── .gitkeep
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Tempest1ss1mo/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the `data/` directory:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── ...
```

### 5. Train Models

**Train Baseline CNN:**
```bash
python src/train.py --model baseline --epochs 50
```

**Train Transfer Learning Model:**
```bash
python src/train.py --model resnet18 --mode finetune --epochs 30
```

### 6. Evaluate Model

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

## 🧠 Model Architecture

### Baseline CNN (LeNet-inspired)

```
Input (48×48×1)
    ↓
Conv2d(32) → ReLU → MaxPool
    ↓
Conv2d(64) → ReLU → MaxPool
    ↓
Conv2d(128) → ReLU → MaxPool
    ↓
Flatten → Dropout(0.5)
    ↓
FC(512) → ReLU → Dropout(0.5)
    ↓
FC(7) → Softmax
```

### Transfer Learning (ResNet18/VGG16)

- **Feature Extraction**: Freeze pretrained layers, train only classifier
- **Fine-tuning**: Unfreeze top layers for domain adaptation

## 📈 Expected Results

| Model | Expected Accuracy |
|-------|------------------|
| Baseline CNN | 60-65% |
| Transfer Learning (Fine-tuned) | 70-75% |

## 🔍 Grad-CAM Visualization

Grad-CAM attention maps highlight which facial regions the model focuses on for predictions:

- Eyes region for surprise/fear
- Mouth region for happy/sad
- Overall facial structure for neutral

## ⚙️ Configuration

Training parameters can be modified in `configs/config.yaml`:

```yaml
model:
  name: resnet18
  pretrained: true
  num_classes: 7

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam

augmentation:
  horizontal_flip: true
  rotation: 10
  zoom_range: 0.1
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- opencv-python
- tqdm
- PyYAML

## 📝 Training Tips

1. **Data Augmentation**: Essential for preventing overfitting
2. **Learning Rate Scheduler**: Use `ReduceLROnPlateau` for better convergence
3. **Class Imbalance**: Consider weighted loss for underrepresented emotions
4. **Early Stopping**: Monitor validation loss to prevent overfitting

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
2. [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
3. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 👤 Author

**Mingliang Yu** (my2899)

---

⭐ Star this repo if you find it helpful!
