# 🌿 Laboratory Work 4 — Custom CNN Image Classifier
### Model Enhancement and Performance Optimization

> A custom Convolutional Neural Network trained on 20 plant species, enhanced through regularization, fine-tuning, and advanced evaluation — achieving an AUC score of **0.9771**, surpassing the Teachable Machine baseline of **0.8963**.

📓 **Google Colab Notebook:** [Open in Colab →](https://colab.research.google.com/drive/1TOvLPOl44HpLeaMng4UooyV_OJktsJKG?usp=sharing)

---

## 📑 Table of Contents

- [A. Model Evaluation Analysis](#-a-model-evaluation-analysis)
- [B. Model Improvement](#-b-model-improvement)
- [C. Performance Comparison](#-c-performance-comparison)
- [D. Explainability — Grad-CAM](#-d-explainability--grad-cam)

---

## 📊 A. Model Evaluation Analysis

---

### 1. What were the weakest-performing classes based on the confusion matrix?

Based on the baseline model's classification report and confusion matrix, the weakest-performing classes were **Barberries, Rubus, Pistacia, Vaccinium, Itea, Actinidia, Abelia, and Magnolia**. These classes had lower F1-scores compared to the other plant classes.

| Class | F1-Score |
|---|---:|
| **Barberries** | 0.3051 |
| **Rubus** | 0.3117 |
| **Pistacia** | 0.3214 |
| **Vaccinium** | 0.3718 |
| **Itea** | 0.3878 |
| **Actinidia** | 0.3964 |
| **Abelia** | 0.4082 |
| **Magnolia** | 0.4118 |

The confusion matrix also showed that some classes were often misclassified as visually similar plants:
- **Vaccinium** was often confused with **Aronia_Melanocarpa**
- **Pieris** was confused with **Abelia**
- **Rubus** was confused with **Vaccinium**

---

### 2. How did Precision, Recall, and F1-score vary across classes?

The precision, recall, and F1-score varied significantly across the 20 plant classes. Some classes performed very well, while others had weaker results.

- **Stronger-performing classes** — Forsythia (`0.8640`), Hibiscus_Syriacus (`0.7750`), Genista, Fuchsia, and Weigela performed well due to their visually distinct features
- **Weaker-performing classes** — Barberries, Rubus, Pistacia, and Vaccinium had much lower F1-scores due to high visual similarity with neighboring classes

> This variation suggests that some plant classes were easier for the CNN to recognize because they had more distinct visual features, while others were more difficult because they looked similar to other classes.

---

### 3. What does a low recall indicate in your model?

> **Low recall** means that the model failed to correctly identify many actual images from a specific class — in other words, the model missed several true samples of that class.

The baseline model's lowest recall scores:

| Class | Recall |
|---|---:|
| **Barberries** | 0.2308 |
| **Rubus** | 0.2500 |
| **Pistacia** | 0.3273 |
| **Ephedra** | 0.3393 |
| **Itea** | 0.3455 |

Many real images from these classes were predicted as other plant classes. Low recall shows that the model was not sensitive enough to recognize all examples of a certain class.

---

### 4. How does AUC score reflect model performance compared to accuracy?

> **Accuracy** measures whether the model's final predicted class is correct. **AUC** measures how well the model separates one class from all other classes using probability scores — even when the top predicted class is wrong.

| Metric | Baseline Model | Improved Model |
|---|---:|---:|
| Validation Accuracy | 0.5265 | **0.7770** |
| AUC Score | 0.9223 | **0.9771** |

The baseline model had a validation accuracy of only **0.5265**, but its corrected softmax-based AUC score was **0.9223**. This means that even though the model did not always choose the correct top class, it was still able to rank the correct class relatively well in many cases.

After improvement, the custom CNN achieved an accuracy of **0.7770** and an AUC score of **0.9771** — showing the improved model became better at both choosing the correct class and separating each plant class from the others.

---

## 🚀 B. Model Improvement

---

### 5. How did data augmentation affect validation accuracy?

Data augmentation helped improve the model's ability to generalize to unseen images. Instead of memorizing the training images, the model learned from slightly modified versions — horizontally flipped, slightly rotated, zoomed, and contrast-adjusted samples.

However, the augmentation was adjusted to be **lighter than the original guide setting** because the previous LW4 model underfitted. The improved setup used:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.10),
])
```

- ✅ Removed `horizontal_and_vertical` flip — vertical flipping creates unnatural plant orientations
- ✅ Reduced rotation, zoom, and contrast values to avoid over-distorting plant features

After applying the improved training pipeline, validation accuracy increased from **0.5265 → 0.7770**, showing that data augmentation combined with the other improvements helped the model perform better on unseen data.

---

### 6. Why is Batch Normalization important in CNNs?

> **Batch Normalization** normalizes the output of convolutional layers, which stabilizes and speeds up the training process by preventing unstable changes in data distribution during training.

In this custom CNN, Batch Normalization was added after each convolutional layer. This:
- Helped the model train more smoothly
- Allowed the deeper CNN architecture to learn better plant features
- Reduced training instability
- Supported better overall validation performance

---

### 7. What role did Dropout play in improving your model?

> **Dropout** randomly disables a percentage of neurons during training, preventing the model from depending too much on specific neurons and encouraging it to learn more general image features.

The high dropout values from the original guide (`0.4` and `0.5`) caused severe underfitting. The improved model used graduated, lighter values:

```text
Dropout layers used: 0.05 → 0.10 → 0.15 → 0.20 → 0.30
```

This was better than using very high dropout values, which previously made the model underfit. The adjusted dropout values helped balance learning and regularization effectively.

---

### 8. How did Early Stopping prevent overfitting?

> **EarlyStopping** monitors the validation loss during training. When the validation loss stops improving for a set number of epochs, training automatically stops and the best model weights are restored.

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

- The model trained until **epoch 59**
- EarlyStopping restored the best weights from **epoch 47**
- This prevented the model from continuing to train after validation loss stopped improving
- The final model used the best validation performance instead of the last training epoch

---

## 📈 C. Performance Comparison

---

### 9. What improvements were observed after modifying the model?

After modifying the custom CNN, all major metrics improved significantly:

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

> ✅ The improved model exceeded the Teachable Machine / previous baseline AUC score of **0.8963**, achieving a final AUC score of **0.9771**.

---

### 10. Which enhancement contributed the most to performance improvement? Why?

The biggest improvement came from the **deeper custom CNN architecture combined with Batch Normalization and learning rate scheduling**:

- **Deeper architecture** — allowed the model to learn more complex plant features
- **Batch Normalization** — stabilized training and prevented gradient instability
- **ReduceLROnPlateau** — lowered the learning rate when validation loss plateaued, helping the model continue improving in later stages

Data augmentation, class weights, Dropout, and EarlyStopping all contributed supporting roles, but the deeper CNN architecture and better training control were the primary reasons for the significant performance gain.

---

### 11. Did the gap between training and validation accuracy decrease? Explain.

The gap did **not decrease** — it slightly increased.

| | Baseline Model | Improved Model |
|---|---:|---:|
| Training Accuracy | 0.6365 | **0.8941** |
| Validation Accuracy | 0.5265 | **0.7770** |
| Generalization Gap | 0.1100 | **0.1171** |

The gap increased slightly from **0.1100 → 0.1171**. However, this is not a bad result because:
- Both training and validation performance improved significantly
- Validation accuracy increased by approximately **25 percentage points**
- Validation loss decreased from **1.5330 → 0.9146**

> This means that although the gap did not decrease, the improved model still generalized much better overall compared to the baseline.

---

## 🔍 D. Explainability — Grad-CAM

---

### 12. How did Grad-CAM help in understanding model predictions?

> **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizes which parts of an input image most influenced the CNN's prediction by producing a heatmap overlay on the original image.

Instead of only seeing the predicted class, Grad-CAM showed the image regions the model focused on when making its decision.

In the sample image:
- **True class:** Hibiscus_Syriacus
- **Predicted class:** Hibiscus_Syriacus ✅
- **Confidence:** **0.91**

The heatmap highlighted the **flower region**, confirming the model was using relevant visual features such as the flower shape, color, and petal structure.

---

### 13. Did the improved model focus on more relevant regions? Provide evidence.

Based on the Grad-CAM result generated in the notebook, the model focused mostly on the **relevant flower region** rather than on the background.

- The selected sample image showed a pink **Hibiscus_Syriacus** flower
- The model correctly predicted the class with **0.91 confidence**
- The heatmap showed stronger activation around the **flower area**, especially around the central and lower-right flower region

> This is evidence that the CNN learned meaningful visual features for that sample.

⚠️ **Note:** The Grad-CAM result shown was generated for the baseline model. To fully prove that the **improved model** focused on more relevant regions, Grad-CAM should also be applied to the improved custom CNN and compared against the baseline Grad-CAM output.

---

### 14. Why is explainability important in real-world AI applications?

Explainability is important because it helps users understand **why** an AI model made a certain prediction. In real-world applications, it is not enough for a model to only give an answer — users also need to know whether the model is focusing on the correct features.

For plant classification, Grad-CAM can show whether the model is focusing on the plant or flower itself instead of unrelated background elements or lighting. This matters because:

| Benefit | Description |
|---|---|
| 🔎 **Transparency** | Users can verify the model is using the right features |
| 🐛 **Debugging** | If the model focuses on backgrounds, the dataset or preprocessing may need fixing |
| 🤝 **Trust** | Stakeholders gain confidence when predictions can be explained |
| ⚖️ **Responsible AI** | Critical for deployment in domains like agriculture, medicine, or security |

> Explainability supports transparency, debugging, trust, and responsible AI use in any real-world system.

---

*Laboratory Work 4 · Custom Image Classifier · TensorFlow / Keras · Google Colab*
