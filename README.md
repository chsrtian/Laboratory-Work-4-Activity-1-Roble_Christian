# 🌿 Laboratory Work 4 — Custom CNN Image Classifier
### Model Enhancement and Performance Optimization

> **A custom Convolutional Neural Network trained on 20 plant species, enhanced through regularization, fine-tuning, and advanced evaluation — achieving an AUC score of 0.9771, surpassing the Teachable Machine baseline of 0.8963.**

📓 **Google Colab Notebook:** [Open in Colab →](https://colab.research.google.com/drive/1TOvLPOl44HpLeaMng4UooyV_OJktsJKG?usp=sharing)

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Model Evaluation Analysis](#-model-evaluation-analysis)
- [Model Improvements](#-model-improvements)
- [Performance Comparison](#-performance-comparison)
- [Explainability — Grad-CAM](#-explainability--grad-cam)
- [Key Takeaways](#-key-takeaways)

---

## 🧭 Executive Summary

This activity focuses on analyzing the weaknesses of a baseline CNN model trained on a 20-class plant species dataset, then systematically applying enhancements — including improved data augmentation, deeper architecture with Batch Normalization, learning rate scheduling, class weighting, and dropout tuning — to produce a significantly improved model that outperforms the Teachable Machine baseline in both accuracy and AUC score.

---

## 📊 Model Evaluation Analysis

### 1. Weakest-Performing Classes

Based on the baseline model's classification report and confusion matrix, the following classes had the lowest F1-scores:

| Class | F1-Score |
|---|---|
| Barberries | 0.3051 |
| Rubus | 0.3117 |
| Pistacia | 0.3214 |
| Vaccinium | 0.3718 |
| Itea | 0.3878 |
| Actinidia | 0.3964 |
| Abelia | 0.4082 |
| Magnolia | 0.4118 |

> 💡 The confusion matrix revealed that visually similar classes were frequently misclassified — for example, **Vaccinium** was often confused with **Aronia_Melanocarpa**, and **Pieris** with **Abelia**.

---

### 2. Precision, Recall, and F1-Score Variation

Performance varied significantly across the 20 plant classes:

- **Stronger classes** — Forsythia (`0.8640`), Hibiscus_Syriacus (`0.7750`), Genista, Fuchsia, and Weigela all performed well due to their visually distinct features
- **Weaker classes** — Barberries, Rubus, Pistacia, and Vaccinium struggled because of high visual similarity to neighboring classes

> This variation indicates that **feature distinctiveness** of each plant class directly impacts how well the CNN can separate them.

---

### 3. What Does Low Recall Indicate?

> **Low recall** means the model failed to correctly identify many actual images from a specific class — in other words, it missed real samples.

The baseline model's lowest recall scores:

| Class | Recall |
|---|---|
| Barberries | 0.2308 |
| Rubus | 0.2500 |
| Pistacia | 0.3273 |
| Ephedra | 0.3393 |
| Itea | 0.3455 |

These classes had many true samples incorrectly predicted as other plant classes, showing the model lacked sufficient sensitivity for them.

---

### 4. AUC Score vs. Accuracy

> **Accuracy** measures whether the model's top predicted class is correct. **AUC** measures how well the model ranks the correct class using probability scores — even when it doesn't always pick the top class correctly.

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Validation Accuracy | 0.5265 | **0.7770** |
| AUC Score | 0.9223 | **0.9771** |

The baseline model had only **52.65%** accuracy but an AUC of **0.9223**, meaning it ranked the correct class well even when it didn't predict it as the top result. The improved model pushed both metrics significantly higher.

---

## 🔧 Model Improvements

### Enhancement 1 — Data Augmentation (Adjusted)

The original guide's augmentation settings were too aggressive and caused underfitting. The settings were lightened to preserve useful visual information while still improving generalization:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.10),
])
```

- ✅ Removed `horizontal_and_vertical` flip — vertical flipping creates unnatural plant orientations
- ✅ Reduced rotation, zoom, and contrast values to avoid excessive distortion
- **Result:** Validation accuracy improved from **0.5265 → 0.7770**

---

### Enhancement 2 — Batch Normalization

> **Batch Normalization** normalizes the output of each convolutional layer, stabilizing and speeding up the training process.

- Added after each `Conv2D` layer in the improved architecture
- Prevented unstable shifts in data distribution during training
- Enabled the deeper CNN to learn plant features more effectively

---

### Enhancement 3 — Dropout Tuning

> **Dropout** randomly disables neurons during training, preventing the model from over-relying on specific features and reducing overfitting.

The high dropout values from the original guide (`0.4` and `0.5`) caused severe underfitting. The improved model used graduated, lighter values:

```text
Dropout layers used: 0.05 → 0.10 → 0.15 → 0.20 → 0.30
```

This balanced regularization with the model's ability to learn, resolving the underfitting problem.

---

### Enhancement 4 — Early Stopping

> **EarlyStopping** monitors validation loss and halts training when it stops improving, then restores the best model weights automatically.

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

- The model trained until **epoch 59**
- Best weights were restored from **epoch 47**
- Prevented unnecessary training after peak validation performance was reached

---

### Enhancement 5 — Learning Rate Scheduling

`ReduceLROnPlateau` was added to automatically reduce the learning rate when validation loss plateaued:

- Allowed the model to fine-tune its weights more carefully in later epochs
- Combined with EarlyStopping, ensured the model used its best-performing state

---

## 📈 Performance Comparison

### Before vs. After — Full Metrics Table

| Metric | Baseline Model | Improved Custom CNN | Difference |
|---|---:|---:|---:|
| Training Accuracy | 0.6365 | **0.8941** | +0.2576 |
| Validation Accuracy | 0.5265 | **0.7770** | +0.2504 |
| Training Loss | 1.2125 | **0.3810** | -0.8314 |
| Validation Loss | 1.5330 | **0.9146** | -0.6184 |
| Macro Precision | 0.5263 | **0.7792** | +0.2529 |
| Macro Recall | 0.5141 | **0.7778** | +0.2637 |
| Macro F1-Score | 0.5103 | **0.7738** | +0.2635 |
| AUC Score | 0.9223 | **0.9771** | +0.0548 |

> ✅ **The improved model achieved an AUC of 0.9771, surpassing the Teachable Machine baseline of 0.8963 by +0.0808.**

---

### Generalization Gap Analysis

| | Baseline | Improved |
|---|---|---|
| Training Accuracy | 0.6365 | 0.8941 |
| Validation Accuracy | 0.5265 | 0.7770 |
| Gap | 0.1100 | 0.1171 |

While the generalization gap increased slightly (+0.0071), this is not a concern — both training and validation performance improved substantially, with validation accuracy rising by **~25 percentage points**.

---

### Top Contributing Enhancement

The biggest performance gain came from the combination of:

1. **Deeper CNN architecture** — allowed the model to learn more complex plant features
2. **Batch Normalization** — stabilized training and prevented gradient instability
3. **ReduceLROnPlateau** — enabled precise fine-tuning in later epochs

Data augmentation, class weights, dropout tuning, and EarlyStopping each contributed supporting roles.

---

## 🔍 Explainability — Grad-CAM

### What is Grad-CAM?

> **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizes which regions of an input image most influenced the CNN's prediction, producing a heatmap overlay on the original image.

### Results

- **Test image:** Hibiscus_Syriacus (pink flower)
- **True class:** Hibiscus_Syriacus
- **Predicted class:** Hibiscus_Syriacus ✅
- **Confidence:** **0.91**

The heatmap showed strongest activation around the **central and lower-right flower region**, confirming the model focused on the actual plant structure — petals, shape, and color — rather than background noise.

### Why Explainability Matters in Real-World AI

- 🔎 **Transparency** — Users can verify the model is using the right features
- 🐛 **Debugging** — If the model focuses on backgrounds, the dataset or preprocessing needs review
- 🤝 **Trust** — Stakeholders gain confidence when predictions can be explained
- ⚖️ **Responsible AI** — Critical for deployment in high-stakes domains like agriculture, medicine, or security

---

## ✅ Key Takeaways

- The baseline CNN suffered from **overfitting** (high train accuracy, low validation accuracy) due to no augmentation or regularization
- The first improvement attempt caused **underfitting** due to over-aggressive augmentation and dropout
- The final improved model resolved both issues through **lighter augmentation, graduated dropout, Batch Normalization, and learning rate scheduling**
- The custom CNN successfully **beat the Teachable Machine AUC baseline** (0.8963) without using any pretrained or transfer learning model
- **Grad-CAM confirmed** the model learned meaningful visual features rather than noise or background artifacts

---

*Laboratory Work 4 — Custom Image Classifier | TensorFlow / Keras | Google Colab*
