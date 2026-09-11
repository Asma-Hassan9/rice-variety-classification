# Rice Variety Classification with CNN and EfficientNetB0

Deep-learning image classification project that identifies five rice varieties from grain images and compares a custom convolutional neural network with EfficientNetB0 transfer learning.

## Project overview

Manual rice inspection can be slow, inconsistent, and dependent on specialist judgement. This project investigates whether computer-vision models can classify rice varieties accurately enough to support automated agricultural quality inspection.

The analysis compares multiple custom CNN configurations with pretrained EfficientNetB0 models. It covers exploratory image analysis, preprocessing, augmentation, stratified splitting, regularization, hyperparameter tuning, transfer learning, fine-tuning, and evaluation.

## Dataset

The project uses the [Rice Image Dataset on Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset), containing 75,000 images distributed across five balanced classes:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

The dataset is not redistributed in this repository. Download it directly from Kaggle and follow its current license and usage terms.

## Objectives

- Explore class balance, image dimensions, brightness, RGB distributions, and image quality.
- Build a custom CNN baseline for five-class rice classification.
- Evaluate regularization and hyperparameter-tuning strategies.
- Apply EfficientNetB0 transfer learning and fine-tuning.
- Compare models using accuracy, precision, recall, F1-score, and confusion matrices.
- Test the selected model on unseen rice images.

## Methodology

The project follows the CRISP-DM framework:

1. Define the agricultural inspection problem.
2. Understand and visually inspect the image dataset.
3. Resize, normalize, encode, augment, and split the images.
4. Train custom CNN and EfficientNetB0 model variants.
5. Tune model architecture and training hyperparameters.
6. Evaluate generalization using a held-out test set.
7. Analyze predictions and model limitations.

## Model comparison

| Model | Reported test accuracy |
|---|---:|
| Baseline CNN | 97.14% |
| Regularized CNN | ~98% |
| Manually tuned CNN | ~98% |
| Rebuilt CNN with selected hyperparameters | 98.68% |
| EfficientNetB0 (frozen backbone) | 99.11% |
| EfficientNetB0 (fine-tuned) | 99.44% |
| Final retrained EfficientNetB0 | **99.58%** |

EfficientNetB0 produced the strongest final performance and more consistent generalization than the custom CNN. The final reported test loss was approximately 0.013.

> Results are based on the dataset split used in the notebook. Performance on images from different cameras, lighting conditions, backgrounds, farms, and processing environments requires independent external validation.

## Technology stack

- Python
- TensorFlow and Keras
- EfficientNetB0
- Keras Tuner
- scikit-learn
- OpenCV
- NumPy and pandas
- Matplotlib and Seaborn
- KaggleHub

## Repository structure

```text
rice-variety-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── rice_variety_classification.ipynb
└── models/
    └── README.md
```

## Running the notebook

1. Create and activate a Python environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `notebooks/rice_variety_classification.ipynb`.
4. Run the dataset-download cell. It uses KaggleHub and does not require credentials to be written inside the notebook.
5. A GPU runtime is recommended for model training and hyperparameter tuning.

## Key findings

- All five classes are balanced, reducing the need for class-weight correction.
- Data augmentation and regularization improved the custom CNN's generalization.
- Transfer learning delivered higher accuracy than training a CNN entirely from scratch.
- Fine-tuning selected EfficientNetB0 layers improved feature adaptation to rice-grain imagery.
- Similar-looking grain shapes still create occasional confusion, especially under variable image conditions.

## Limitations

- The images come from a controlled dataset and may not represent real production environments.
- Random image splitting can produce optimistic estimates when acquisition conditions are highly similar.
- External validation across cameras, backgrounds, lighting, and geographic sources was limited.
- Hyperparameter tuning is computationally expensive.

## Future work

- Evaluate the model on independently collected rice images.
- Use group-aware splitting when acquisition batch identifiers are available.
- Add Grad-CAM visualizations to explain model attention.
- Compare MobileNetV3 and newer EfficientNet variants.
- Export a lightweight model for mobile or edge deployment.
- Build a simple prediction interface for quality inspectors.

## Author

**Asma Abdiwali Hassan**  
MSc Data Science and Business Analytics  
[LinkedIn](https://www.linkedin.com/in/asma-hassan-431a37368)

## Responsible use

This project is an academic prototype. Predictions should be reviewed by qualified personnel before being used for agricultural grading, pricing, or trade decisions.

