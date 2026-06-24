# Bearing Fault Detection using Machine Learning

## Overview
This project presents a machine learning-based approach for **bearing fault detection and classification** using vibration signal data. The system processes bearing datasets, performs exploratory data analysis (EDA), trains multiple classification models, and compares their performance to identify the most effective approach for fault diagnosis.

The project also includes severity analysis, saved trained models, evaluation visualizations, and dashboard files for prediction and analysis.

---

## Objectives
- Detect faults in bearing vibration data.
- Classify different bearing operating conditions.
- Train and evaluate multiple machine learning models.
- Compare model performance using standard metrics.
- Perform fault severity analysis.
- Enable prediction using saved trained models.

---

## Dataset
The project uses vibration signal datasets representing different bearing conditions.

### Bearing Conditions Included
- Healthy Bearing
- Inner Race Fault (IR)
- Outer Race Fault (OR)
- Multiple fault severity conditions

Dataset files:
- `B007_3.csv`
- `IR007_3.csv`
- `OR007@3_3.csv`
- `OR007@6_3.csv`
- `OR007@12_3.csv`
- `combined_bearing_dataset.csv`

---

## Project Workflow

### 1. Data Preprocessing
- Data loading and cleaning
- Dataset merging
- Feature extraction
- Scaling and normalization

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis
- Fault visualization
- Statistical understanding of vibration signals

### 3. Model Development
Implemented machine learning models:

- Random Forest
- K-Nearest Neighbors (KNN)
- Additional classification models for comparison

Saved models:
- `random_forest_model.pkl`
- `knn_model.pkl`
- `severity_model.pkl`
- `scaler.pkl`

### 4. Model Evaluation & Comparison
Models were trained, evaluated, and compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Visual comparison of model performance was performed to identify the most accurate model for bearing fault classification.

Evaluation outputs:
- `fig_evaluation.png`
- `fig_evaluation_all_models.png`
- `fig_models_large.png`
- `fig_threshold_zoomed.png`

### 5. Severity Analysis
Severity prediction was implemented to analyze the extent of bearing faults.

Notebook:
- `severity_analysis.ipynb`

---

## Dashboard
Interactive dashboards were developed for visualization and prediction.

Files:
- `dashboard1.py`

Run:

```bash
python dashboard1.py
```

---

## Repository Structure

```
Bearing_Fault_Detection/
│
├── bearingfault_modelcomparison.ipynb
├── fault_classification.ipynb
├── severity_analysis.ipynb
│
├── B007_3.csv
├── IR007_3.csv
├── OR007@3_3.csv
├── OR007@6_3.csv
├── OR007@12_3.csv
├── combined_bearing_dataset.csv
│
├── dashboard1.py
│
├── random_forest_model.pkl
├── knn_model.pkl
├── severity_model.pkl
├── scaler.pkl
│
├── fig_evaluation.png
├── fig_evaluation_all_models.png
├── fig_models_large.png
├── fig_threshold_zoomed.png
│
├── proper_blind_test_files.zip
├── test_file_1.csv
├── test_file_2.csv
├── test_file_3.csv
├── test_file_4.csv
├── test_file_5.csv
│
└── README.md
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/kvbd176/Bearing-Fault-Detection.git
```

Move into directory:

```bash
cd Bearing-Fault-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch notebook:

```bash
jupyter notebook
```

Run dashboard:

```bash
python dashboard1.py
```

---

## Results
- Built a complete bearing fault classification pipeline.
- Trained multiple machine learning models.
- Performed comparative analysis across models.
- Identified the best-performing classifier.
- Implemented severity prediction and dashboard visualization.

---

## Future Improvements
- Deep learning-based fault diagnosis
- Real-time monitoring integration
- Advanced feature extraction
- Hyperparameter optimization

---

## Author & Acknowledgement

**Author:**  
Venkata Durga Bhavani Kalla

**Acknowledgement:**  
This project was developed by **Venkata Durga Bhavani Kalla** under the supervision and guidance of **Prabas Banerjee**, currently serving as a Research Scientist at Experiqs Pvt. Ltd. Their guidance and support contributed significantly to the successful completion of this project.
