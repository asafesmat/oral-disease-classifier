# Oral Disease Image Classifier

A deep learning pipeline that classifies oral images into six conditions, with a Streamlit web app for inference. Built with TensorFlow/Keras as part of a course project.

> **Disclaimer:** Educational project only. It is not a medical device and must not be used for diagnosis. Consult a dentist.

## Classes

Calculus, Caries, Discoloration, Gingivitis, Mouth Ulcer, Hypodontia.

## Dataset

[Oral Diseases (Kaggle)](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)

Only the original image folders were used. The pre-augmented folders and the YOLO-annotated folder were excluded, because they contain modified copies of the same images and would leak between splits.

Data cleaning (exact file hashing):

- 5,563 original images were collected.
- Exact duplicate files were reduced to one copy.
- Images that appeared under more than one class (1,510 files, mostly Calculus/Gingivitis overlaps) were removed because they have no single valid label.
- 3,507 images remained.

Split: stratified 70/15/15 with a fixed seed (train 2,454 / validation 526 / test 527). No file hash is shared between splits.

## Method

- Input: images resized to 224x224. Preprocessing for ResNet50 is part of the model.
- Augmentation (training set only): horizontal flip, small rotation, zoom.
- Class imbalance handled with balanced class weights computed from the training set.
- Training: Adam optimizer, early stopping and learning-rate reduction on validation loss.
- Models compared: a custom CNN trained from scratch, and ImageNet-pretrained ResNet50, EfficientNetB0 and DenseNet121.
- ResNet50 setup: ImageNet weights, the last 30 layers of the base network unfrozen, dropout 0.3, Adam with a learning rate of 1e-5.

## Results (test set, 527 images)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| ResNet50 | 0.9583 | 0.9336 | 0.9082 | 0.9172 |
| EfficientNetB0 | 0.8880 | 0.8364 | 0.8466 | 0.8406 |
| DenseNet121 | 0.8121 | 0.7270 | 0.7702 | 0.7383 |
| Custom CNN | 0.7287 | 0.6240 | 0.6767 | 0.6363 |

ResNet50 was deployed. Its per-class results:

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Calculus | 0.940 | 0.940 | 0.940 | 67 |
| Caries | 0.952 | 0.690 | 0.800 | 29 |
| Discoloration | 0.815 | 0.846 | 0.830 | 26 |
| Gingivitis | 0.973 | 0.978 | 0.976 | 186 |
| Mouth Ulcer | 0.947 | 1.000 | 0.973 | 36 |
| Hypodontia | 0.973 | 0.995 | 0.984 | 183 |

## Limitations

- The best model was selected using test-set scores, so the reported test numbers are likely optimistic. A cleaner protocol would select on validation data and evaluate on the test set once.
- The source images include visibly pre-augmented samples (rotated, black corners). Modified copies of the same photo may exist across splits and are not caught by exact hashing, so real-world performance may be lower.
- Caries has the lowest recall (0.69): about three in ten caries images in the test set were missed.
- The smallest classes have only 26-29 test images, so their metrics are noisy.
- The model fits the training set almost perfectly (99.9% accuracy vs 95.8% on the test set), which indicates overfitting.
- Single public dataset; no external validation.

## Run the app locally

The trained model (~210 MB) is not stored in this repository because of GitHub's file-size limit. Place `best_oral_disease_model.keras` next to `app.py`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tested with Python 3.12, TensorFlow 2.20.0, Keras 3.13.2 and Streamlit 1.66.1.

## Project structure

- `app.py`: Streamlit inference app
- `requirements.txt`: pinned dependencies
- `.gitignore`: excludes the environment and model files