# 🫁 Lung Cancer Classification Using GoogLeNet

> **Deep Learning | Computer Vision | Transfer Learning | Medical Image Classification | PyTorch**

A deep learning project for **multi-class lung cancer image classification** using **GoogLeNet (Inception)** with transfer learning and staged fine-tuning.

The model is trained to classify lung tissue images into three classes:

* `lung_aca`
* `lung_n`
* `lung_scc`

The implementation uses **PyTorch and TorchVision**, with ImageNet-pretrained GoogLeNet as the backbone. The project includes data augmentation, transfer learning, auxiliary classifier optimization, fine-tuning, learning-rate scheduling, model checkpointing, classification-report evaluation, confusion-matrix analysis, and single-image inference.

---

## 📌 Project Overview

Lung cancer image classification is a challenging computer vision problem because different tissue patterns can have visually similar characteristics.

This project explores a deep learning pipeline that learns discriminative visual representations from lung images and predicts the corresponding class.

Instead of training GoogLeNet completely from scratch, the project follows a **transfer learning strategy**:

```text
Lung Images
     ↓
Data Augmentation
     ↓
ImageNet Pretrained GoogLeNet
     ↓
Freeze Backbone
     ↓
Train Classification Layers
     ↓
Load Best Checkpoint
     ↓
Fine-Tune Deeper Inception Layers
     ↓
Learning Rate Scheduling
     ↓
Best Model
     ↓
Test Evaluation
     ↓
Single Image Prediction
```

---

## 🎯 Objective

The primary objective is to build a reusable deep learning pipeline capable of classifying lung tissue images into three categories using a pretrained GoogLeNet architecture.

### Classification Classes

| Class      | Description             |
| ---------- | ----------------------- |
| `lung_aca` | Adenocarcinoma          |
| `lung_n`   | Normal lung tissue      |
| `lung_scc` | Squamous cell carcinoma |

> **Note:** The model is intended as a machine-learning/computer-vision project and should not be interpreted as a clinical diagnostic system.

---

## 🚀 Key Features

* Transfer learning using **ImageNet-pretrained GoogLeNet**
* Three-class lung image classification
* Training / validation / testing pipeline
* Image augmentation for training data
* ImageNet normalization
* GPU/CPU device detection
* Frozen-backbone training stage
* GoogLeNet auxiliary classifier training
* Fine-tuning of deeper Inception blocks
* Adam optimizer
* Weight decay regularization
* ReduceLROnPlateau learning-rate scheduler
* Best-model checkpointing
* Classification report
* Confusion matrix
* Single-image inference
* Prediction confidence calculation
* PyTorch-based implementation

---

## 🧠 Model Architecture

The project uses **GoogLeNet**, also known as the **Inception architecture**, from TorchVision.

```python
from torchvision.models import googlenet, GoogLeNet_Weights

model = googlenet(
    weights=GoogLeNet_Weights.DEFAULT,
    aux_logits=True
)
```

The pretrained ImageNet model is adapted for the project's three-class classification problem.

The original final classification layer is replaced:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)
```

The auxiliary classifiers are also modified:

```python
model.aux1.fc2 = nn.Linear(
    model.aux1.fc2.in_features,
    num_classes
)

model.aux2.fc2 = nn.Linear(
    model.aux2.fc2.in_features,
    num_classes
)
```

---

# 🔬 Transfer Learning Strategy

The project uses a **two-stage training strategy**.

## Stage 1 — Frozen Backbone

Initially, the pretrained GoogLeNet parameters are frozen:

```python
for param in model.parameters():
    param.requires_grad = False
```

Only the newly configured classification layers and auxiliary classifier layers are trained.

This allows the model to adapt its final representation to the lung-image classification task without immediately modifying the pretrained feature extractor.

### Stage 1 Optimizer

```python
optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.001,
    weight_decay=1e-4
)
```

---

# 🔧 Stage 2 — Fine-Tuning

After the first training stage, the best validation checkpoint is loaded.

The deeper GoogLeNet Inception blocks are then unfrozen:

```python
for param in model.inception5a.parameters():
    param.requires_grad = True

for param in model.inception5b.parameters():
    param.requires_grad = True
```

The classifier layers remain trainable.

A much smaller learning rate is then used:

```python
lr = 1e-5
```

This helps make smaller updates to the pretrained feature representations during fine-tuning.

---

# 🧮 GoogLeNet Auxiliary Loss

One of the interesting aspects of this implementation is that it explicitly handles GoogLeNet's auxiliary outputs during training.

The training loss is calculated as:

```text
Total Loss =
Main Loss
+ 0.3 × Auxiliary Loss 1
+ 0.3 × Auxiliary Loss 2
```

This is implemented as:

```python
loss = (
    loss_main
    + 0.3 * loss_aux1
    + 0.3 * loss_aux2
)
```

The auxiliary classifiers provide additional supervision during training and are part of the original GoogLeNet design.

During validation and inference, only the main output is used.

---

# 🖼️ Data Preprocessing

The project uses separate transformations for training and validation/testing.

## Training Augmentation

The training pipeline includes:

* Resize
* Random horizontal flip
* Random rotation
* Random resized crop
* Color jitter
* Tensor conversion
* ImageNet normalization

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])
```

### Why augmentation?

Medical image datasets can be limited in size. Augmentation helps expose the model to variations in the training images and can improve generalization.

---

## Validation and Test Preprocessing

Validation and test images are processed without random augmentation:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])
```

This ensures evaluation is performed on deterministic preprocessing rather than randomly modified images.

---

# 📂 Dataset Structure

The notebook expects the dataset to be organized using `ImageFolder`:

```text
split_dataset/
│
├── train/
│   ├── lung_aca/
│   ├── lung_n/
│   └── lung_scc/
│
├── val/
│   ├── lung_aca/
│   ├── lung_n/
│   └── lung_scc/
│
└── test/
    ├── lung_aca/
    ├── lung_n/
    └── lung_scc/
```

This structure allows PyTorch's `ImageFolder` dataset loader to automatically assign class labels based on directory names.

---

# ⚙️ Data Loading

The project uses:

```python
from torchvision import datasets
```

and loads the dataset using:

```python
datasets.ImageFolder(...)
```

Data is then provided to the model through PyTorch `DataLoader`.

```python
BATCH_SIZE = 32
```

Training data is shuffled, while validation and testing data are not shuffled.

---

# 🖥️ Hardware Acceleration

The project automatically detects whether CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

If CUDA is available, the GPU name is also displayed.

This allows the same implementation to run on:

* CPU
* NVIDIA CUDA GPU

---

# 🏋️ Training Pipeline

The model is initially trained for:

```python
EPOCHS = 5
```

The training loop records:

* Training loss
* Training accuracy
* Validation loss
* Validation accuracy

A checkpoint is saved whenever validation accuracy improves.

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc

    torch.save(
        model.state_dict(),
        "googlenet_stage1_best.pth"
    )
```

---

# 🔄 Fine-Tuning Pipeline

After Stage 1:

```text
Best Stage-1 Model
        ↓
Load checkpoint
        ↓
Unfreeze inception5a
        ↓
Unfreeze inception5b
        ↓
Keep classifiers trainable
        ↓
Lower learning rate
        ↓
ReduceLROnPlateau
        ↓
Fine-tune
```

The notebook performs up to:

```python
FINE_TUNE_EPOCHS = 10
```

A learning-rate scheduler is used:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)
```

The scheduler reduces the learning rate when validation loss stops improving.

---

# 📊 Model Evaluation

The final model is evaluated using the test dataset.

The project generates a classification report:

```python
classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes
)
```

This provides class-level evaluation metrics such as:

* Precision
* Recall
* F1-score
* Support

This is more informative than reporting accuracy alone, particularly for multi-class medical image classification.

---

# 🔲 Confusion Matrix

A confusion matrix is generated to analyze class-level prediction behavior:

```python
cm = confusion_matrix(
    all_labels,
    all_predictions
)
```

The matrix helps identify:

* Correct predictions
* Misclassified samples
* Which classes are confused with each other
* Class-specific model weaknesses

---

# 🔍 Single Image Prediction

The trained model can also predict an individual image.

The inference pipeline is:

```text
Input Image
    ↓
RGB Conversion
    ↓
Validation Transform
    ↓
Tensor
    ↓
Batch Dimension
    ↓
GoogLeNet
    ↓
Softmax
    ↓
Predicted Class
    ↓
Confidence Score
```

The model calculates class probabilities using:

```python
probabilities = torch.softmax(
    output,
    dim=1
)
```

The predicted class is selected using:

```python
predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()
```

The corresponding probability is reported as the prediction confidence.

---

# 💾 Model Artifacts

The notebook creates PyTorch model checkpoints:

```text
googlenet_stage1_best.pth
best_googlenet_finetuned.pth
lung_cancer_googlenet.pth
```

The final model is saved using:

```python
torch.save(
    model.state_dict(),
    "lung_cancer_googlenet.pth"
)
```

---

# 🛠️ Tech Stack

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Core programming                    |
| PyTorch      | Deep learning framework             |
| TorchVision  | Computer vision models and datasets |
| GoogLeNet    | CNN architecture                    |
| NumPy        | Numerical operations                |
| Matplotlib   | Visualization                       |
| scikit-learn | Model evaluation                    |
| PIL          | Image processing                    |
| Google Colab | Development/training environment    |
| CUDA         | GPU acceleration when available     |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required packages:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

For GPU training, install the appropriate PyTorch version for your CUDA environment.

---

# ▶️ Running the Project

## 1. Prepare the Dataset

Create the following directory structure:

```text
split_dataset/
├── train/
├── val/
└── test/
```

Each directory should contain:

```text
lung_aca/
lung_n/
lung_scc/
```

## 2. Update Dataset Path

Change the dataset path in the notebook:

```python
DATA_DIR = "path/to/split_dataset"
```

## 3. Run the Notebook

Open:

```text
Lung_cancer_prediction using Googlenet.ipynb
```

Run the cells sequentially.

The notebook will:

1. Detect available hardware
2. Load datasets
3. Apply transformations
4. Create DataLoaders
5. Load pretrained GoogLeNet
6. Replace classification layers
7. Train the classifier
8. Save the best Stage-1 model
9. Fine-tune deeper layers
10. Evaluate the final model
11. Generate classification metrics
12. Generate a confusion matrix
13. Save the final model
14. Perform single-image prediction

---

# 📈 Project Workflow

```text
                ┌──────────────────────┐
                │    Lung Images       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Dataset Organization │
                │ train / val / test   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Image Augmentation   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Pretrained GoogLeNet │
                │     ImageNet         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Freeze Backbone      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Train Classifiers    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Best Checkpoint      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Fine-Tune Inception  │
                │ 5a + 5b Layers       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Test Evaluation      │
                └──────────┬───────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       Classification Report   Confusion Matrix
                │
                ▼
        Single Image Inference
```

---

# 🧪 Engineering Decisions

### Why Transfer Learning?

Training a deep CNN from scratch generally requires substantial amounts of labeled data and computational resources.

Using ImageNet-pretrained GoogLeNet provides a strong initialization and allows the project to focus on adapting learned visual representations to the target classification task.

### Why Freeze the Backbone Initially?

Freezing the pretrained layers reduces the number of trainable parameters during the initial adaptation stage and helps stabilize training of the new classification layers.

### Why Fine-Tune Later?

After the classification layers adapt to the target dataset, deeper feature-extraction layers can be fine-tuned with a smaller learning rate to specialize the pretrained representation.

### Why Use a Lower Learning Rate During Fine-Tuning?

A learning rate of `1e-5` is used during fine-tuning to make smaller parameter updates and reduce the risk of destroying useful pretrained representations.

---

# 📁 Recommended Repository Structure

For a professional GitHub repository, the project can be organized as:

```text
lung-cancer-googlenet/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── Lung_cancer_prediction_using_GoogLeNet.ipynb
│
├── models/
│   └── lung_cancer_googlenet.pth
│
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

> The current uploaded notebook is notebook-based. The `src/` structure above is a recommended production-oriented organization for a future refactor.

---

# 🔐 Important Note About the Dataset

The notebook uses a local/Google Drive dataset path and does not include the image dataset itself.

Do **not** commit a large medical image dataset or unnecessary model binaries directly to GitHub.

Instead, document:

* Dataset name/source
* Dataset structure
* Number of classes
* Train/validation/test split
* Preprocessing
* How to obtain the dataset

If the dataset has licensing or usage restrictions, follow the original dataset's terms.

---

# ⚠️ Medical AI Disclaimer

This project is intended for **educational, research, and machine-learning engineering purposes**.

The model should **not** be used as a standalone medical diagnostic system or as a replacement for evaluation by qualified medical professionals.

Performance on a test dataset does not automatically imply clinical validity, robustness, fairness, or generalization to other hospitals, scanners, populations, or acquisition protocols.

---

# 📌 Limitations

The current implementation has several areas that can be improved:

* Dataset size and diversity may limit generalization.
* The notebook uses a fixed three-class classification setup.
* No external validation dataset is included.
* No clinical deployment pipeline is implemented.
* Model calibration is not evaluated.
* Explainability methods such as Grad-CAM are not currently included.
* The notebook does not establish clinical-grade performance.
* Reproducibility could be improved by explicitly controlling random seeds.
* Hyperparameter optimization is not implemented.

These limitations are important when moving from an experimental notebook toward a production or research-grade medical AI system.

---

# 🚀 Future Improvements

The project can be extended into a more production-oriented AI system by adding:

### Model Improvements

* Compare GoogLeNet with ResNet, DenseNet, EfficientNet and ConvNeXt
* Hyperparameter optimization
* More systematic fine-tuning
* Class-weighted loss if class imbalance exists
* Label smoothing
* Learning-rate warmup
* Mixed-precision training

### Explainable AI

Add:

* Grad-CAM
* Grad-CAM++
* Saliency maps
* Feature visualization

This would help visualize which image regions influenced model predictions.

### Evaluation

Add:

* ROC-AUC
* Precision-Recall curves
* Per-class sensitivity
* Specificity
* Calibration curves
* Cross-validation
* External validation

### Productionization

The trained model could be integrated into:

```text
Frontend
   ↓
API
   ↓
Image Validation
   ↓
Preprocessing
   ↓
GoogLeNet Inference
   ↓
Prediction + Confidence
   ↓
Response
```

Potential deployment technologies include FastAPI, Docker and cloud inference services.

---

# 🧑‍💻 AI Engineering Perspective

This project demonstrates several skills relevant to an **AI Engineer / Machine Learning Engineer** role:

### Machine Learning

* Supervised learning
* Multi-class classification
* Model evaluation
* Generalization

### Deep Learning

* CNN architectures
* Transfer learning
* Fine-tuning
* Auxiliary classifiers
* Loss functions
* Optimizers
* Learning-rate scheduling

### Computer Vision

* Image preprocessing
* Data augmentation
* Image classification
* Tensor-based image inference

### Model Engineering

* Checkpointing
* Train/validation monitoring
* Model serialization
* GPU acceleration
* Inference pipeline

### Evaluation

* Classification report
* Confusion matrix
* Prediction confidence

---

# 💡 What Makes This Project Different

This is not simply:

```text
Dataset → CNN → Accuracy
```

The implementation follows a more realistic deep-learning workflow:

```text
Pretrained Model
       ↓
Transfer Learning
       ↓
Frozen Feature Extractor
       ↓
Task-Specific Classifier
       ↓
Best Checkpoint
       ↓
Selective Fine-Tuning
       ↓
Learning Rate Scheduling
       ↓
Model Evaluation
       ↓
Inference
```

This demonstrates an understanding of **how pretrained deep-learning models can be adapted and optimized for a specialized computer-vision problem**.

---

# 📊 Results

The notebook generates the following evaluation outputs:

* Training accuracy curve
* Validation accuracy curve
* Classification report
* Confusion matrix
* Single-image prediction
* Prediction confidence

Actual numerical results should be added here after running the final notebook and recording the final test metrics.

Example format:

```text
Test Accuracy: XX.XX%

Class-wise Performance:

Class        Precision    Recall    F1-Score
------------------------------------------------
lung_aca       XX.XX%      XX.XX%     XX.XX%
lung_n        XX.XX%      XX.XX%     XX.XX%
lung_scc      XX.XX%      XX.XX%     XX.XX%
```

---

# 📚 Core Concepts Demonstrated

```text
Transfer Learning
        │
        ├── ImageNet Pretraining
        ├── Frozen Backbone
        ├── Classifier Adaptation
        └── Fine-Tuning
                │
                ▼
          GoogLeNet
                │
        ┌───────┴───────┐
        ▼               ▼
    Main Output     Auxiliary Outputs
        │               │
        └───────┬───────┘
                ▼
         Cross Entropy Loss
                │
                ▼
          Adam Optimizer
                │
                ▼
      Learning Rate Scheduler
                │
                ▼
        Best Model Selection
                │
                ▼
       Test Set Evaluation
```

---

# 👨‍💻 Author

**Thilaksri**

AI / Data Science Trainer
Interested in Machine Learning, Deep Learning, Computer Vision and AI Engineering.

---

# ⭐ Project Highlights

```text
✓ PyTorch
✓ GoogLeNet / Inception
✓ Transfer Learning
✓ Fine-Tuning
✓ Computer Vision
✓ Medical Image Classification
✓ Data Augmentation
✓ Auxiliary Classifiers
✓ Adam Optimizer
✓ Learning Rate Scheduling
✓ Model Checkpointing
✓ Confusion Matrix
✓ Classification Report
✓ Single Image Inference
✓ GPU Support
```

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and exploring the implementation.

---

### Disclaimer

This repository represents an educational/research implementation of a deep-learning image-classification workflow. Predictions should not be interpreted as medical diagnoses or used to make clinical decisions without appropriate validation, regulatory approval, and professional oversight.
