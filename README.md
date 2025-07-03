# EEG Motor Imagery Classification

This project compares and applies two feature extraction techniques, Region-Level Feature Extraction and Common Spatial Patterns (CSP), to classify Electroencephalography (EEG) motor imagery tasks. It aims to classify different motor imagery tasks (Left Hand, Right Hand, Feet, Tongue) using a self-developed deep learning model with attention and residual connections.

## Project Structure

The work is split into several Jupyter notebooks, each addressing a specific stage of the EEG signal processing and classification workflow:

1.  `1. Preprocessing.ipynb`: Carries out initial data loading, filtering (bandpass and notch), artifact removal through ICA, and creation of overlapping epochs.
2.  `2-A. Region-Level Feature Extraction.ipynb`: Implements and uses region-level feature extraction by computing hemispheric differences and adding them to midline channel signals.
3.  `2-B. CSP Feature Extraction.ipynb`: Implements and applies Common Spatial Patterns (CSP) for spatial filtering.
4.  `3. Augmentation.ipynb`: Contains the `EEGAugmentation` class for applying various data augmentation techniques on the EEG data.
5.  `4. Model + Training.ipynb`: Defines the `StateOfTheArtEEGNet` model architecture, including attention and residual connections, configures the training pipeline with several loss functions (Cross-Entropy, Focal Loss, Label Smoothing Loss) and advanced training techniques like mixed precision and early stopping.
6.  `5. Evaluation.ipynb`: Includes complete evaluation of the trained model, classification report, confusion matrix, class-wise accuracy, prediction confidence distribution along with ROC curves with AUC scores.

## Feature Extraction Methods

### Region-Level Feature Extraction

It is a feature extraction based on predefined anatomical regions of the brain. It calculates the difference between the corresponding channels of the left and right hemispheres and combines these differences with the signals from the midline channels.

* **Hemispheric Difference:** For each pair of corresponding left and right hemisphere channels ($j \in [1, L_g]$), the difference $d_j = s_i^l - s_j^r$ is computed, where $s_i^l$ and $s_j^r$ are left and right hemisphere channel signals, respectively.
* **Final Feature Vector:** The set of all hemispheric differences ($D = [d_j]$) is concatenated with the midline signals ($s^m$), forming the final feature vector $X = [D^T s^{m T}]$.

### Common Spatial Patterns (CSP)

CSP is a widely used spatial filtering technique for the discrimination of two classes of EEG signals. It learns spatial filters that maximize the variance of one class while minimizing the variance of the other.

## Model

The centerpiece of the classification system is a custom deep learning model. The model starts with the conventional EEGNet architecture and extends it with:

* **Multi-Head Attention:** Allows the model to attend to different positions of the input sequence simultaneously.
* **Channel Attention:** Dynamically re-weights channel-wise features with a greater emphasis on more significant channels.
* **Spatial Attention:** Focuses attention on salient spatial points in the EEG signals.
* **Residual Connections:** Facilitate training deeper networks by making gradients easier to flow.

The training includes:
* **Multi-Loss Functions:** Cross-Entropy Loss, Focal Loss, and Label Smoothing Loss are blended together to handle potential issues like class imbalance and label noise.
* **Optimized Training:** `AdamW` optimizer with `CosineAnnealingWarmRestarts` learning rate scheduler, gradient clipping, and mixed-precision training for fast and stable convergence.
* **Early Stopping:** Prevents overfitting by monitoring validation accuracy and stopping training when the improvement plateaus.
* **Weighted Random Sampler:** Addresses class imbalance in the training data by assigning greater weight to under-represented classes.

## Results

The project compared the model's performance with features extracted by **Region-Level Feature Extraction** and **Common Spatial Patterns (CSP)**.

* **Region-Level Feature Extraction Accuracy:** Approximately **90%**
* **Common Spatial Patterns (CSP) Accuracy:** Approximately **95%**

These results indicate that CSP provides a more discriminative feature representation for this motor imagery classification task than region-level features, with higher classification accuracy.

## Setup and Usage

### Prerequisites

* Python 3.x
* `mne`
* `numpy`
* `scipy`
* `torch`
* `scikit-learn`
* `matplotlib`
* `seaborn`

### Installation

Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/EEG-Motor-Imagery-Classification.git](https://github.com/yourusername/EEG-Motor-Imagery-Classification.git)
    cd EEG-Motor-Imagery-Classification
    ```
### Data

The project uses data from the BCI Competition IV 2a dataset. You need to download and place the raw `.gdf` files and their true corresponding labels (`.mat` files) in the given `raw_data_folder` and `mat_folder` paths in `1. Preprocessing.ipynb`.

### Running the Notebooks

Execute the Jupyter notebooks in the following order:

1.  **`1. Preprocessing.ipynb`**: Run all cells to preprocess raw EEG data. This will save cleaned data in `cleaned_data_folder_1` and `cleaned_data_folder_2`.
2.  Choose **one** of the feature extraction notebooks:
    * **`2-A. Region-Level Feature Extraction.ipynb`**: Run all cells for region-level feature extraction.
    * **`2-B. CSP Feature Extraction.ipynb`**: Run all cells for CSP feature extraction.
*(Run only one of the following before proceeding, as they produce `X_first_session`, `X_second_session`, `y_train`, and `y_test` that are used in subsequent notebooks.)*
3.  **`3. Augmentation.ipynb`**: Run all cells to augment the data from the first session. This creates `X_train_combined` and `y_train_combined`, and finally the `X` and `y` datasets to split for.
4.  **`4. Model + Training.ipynb`**: Run all cells to define, initialize, and train the model. Splitting of data and data loader creation is also performed in this notebook.
5.  **`5. Evaluation.ipynb`**: Run all cells to evaluate the model and generate performance metrics and visualizations.

## Key Features

* **Robust Preprocessing:** Filtering, EOG artifact removal via ICA, and robust normalization are featured.
* **Advanced Feature Extraction:** Region-Level features and CSP are compared.
* **Comprehensive Data Augmentation:** Various techniques like Gaussian noise, time shifting, amplitude scaling, frequency shifting, channel dropout, Mixup, and time warping.
* **EEGNet Architecture with New Features:** Multi-Head, Channel, and Spatial Attention, as well as Residual Blocks.
* **Sophisticated Training Regime:** Employs more than one loss function, dynamic learning rate scheduling, mixed precision, and early stopping.
* **Detailed Evaluation:** Provides accuracy, classification report, confusion matrix, class-wise accuracy, prediction confidence, and ROC curves.

## License

This project is open source and available under the [MIT License](LICENSE).

