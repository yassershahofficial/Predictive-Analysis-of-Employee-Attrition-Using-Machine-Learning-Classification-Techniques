# Predictive Analysis of Employee Attrition Using Machine Learning Classification Techniques

## Overview

This project implements a machine learning solution to predict employee attrition using various classification algorithms. The goal is to identify employees who are likely to leave the organization, enabling proactive retention strategies.

The project performs comprehensive data analysis, feature engineering, and evaluates multiple machine learning models including:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

## Project Structure

```
.
├── classifications.ipynb      # Main Jupyter notebook with complete analysis
├── employee_data.csv          # Employee dataset (3000 records)
├── eda_plots.png             # Exploratory Data Analysis visualizations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

- **Data Preprocessing**: Handles missing values, removes data leakage, and creates derived features
- **Feature Engineering**: Calculates tenure, handles categorical encoding, and scales numerical features
- **Exploratory Data Analysis**: Visualizes attrition patterns, department analysis, and correlation matrices
- **Model Comparison**: Evaluates multiple classification algorithms with comprehensive metrics
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC scores

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

6. **Open the notebook**
   - Navigate to `classifications.ipynb` in the Jupyter interface
   - Run all cells sequentially (Cell → Run All)

## Data Description

The dataset contains employee information with the following key features:
- **Demographics**: Gender, Race, Marital Status, State
- **Employment Details**: Department, Division, Job Function, Employee Type
- **Performance Metrics**: Performance Score, Current Employee Rating
- **Employment History**: Start Date, Exit Date, Tenure
- **Target Variable**: Attrition Flag (0 = Stay, 1 = Left)

**Dataset Source**: The dataset was obtained from [Kaggle - Employee Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/employeedataset/data).

**Note**: The dataset is imbalanced with approximately 86.8% of employees staying and 13.2% leaving.

## Expected Outputs

### 1. Data Processing Outputs
- **Cleaned Dataset**: Original 3000 records reduced to 2931 after removing "Future Start" employees
- **Feature Engineering**: 17 features prepared for modeling after removing identifiers and leakage variables
- **Train-Test Split**: 80-20 split (2344 training, 587 test samples) with stratification

### 2. Visualizations

The notebook generates several visualizations:

- **EDA Plots** (`eda_plots.png`):
  - Target distribution (attrition count)
  - Attrition by department
  - Performance score vs. attrition
  - Correlation heatmap of numerical features

- **Model Evaluation Plots**:
  - Confusion matrix for the best model
  - ROC curve showing model performance
  - Feature importance plot (for tree-based models)

### 3. Model Performance Metrics

The notebook outputs a comparison table with the following metrics for each model:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | ~XX% | ~XX% | ~XX% | ~XX% | ~0.XX |
| Random Forest | ~XX% | ~XX% | ~XX% | ~XX% | ~0.XX |
| Gradient Boosting | ~XX% | ~XX% | ~XX% | ~XX% | ~0.XX |

**Note**: Actual values will vary based on the dataset. The models are sorted by Recall score, as identifying employees likely to leave (True Positives) is prioritized over predicting those who will stay.

### 4. Console Outputs

- Data shape information at each processing step
- Missing value counts
- Feature lists (numeric and categorical)
- Training progress for each model
- Performance metrics for each algorithm
- Model comparison leaderboard

## Key Insights

1. **Class Imbalance**: The dataset shows significant class imbalance (86.8% vs 13.2%), which is addressed using stratified sampling and class weighting.

2. **Data Leakage Prevention**: Features like `ExitDate`, `TerminationType`, and `TerminationDescription` are removed to prevent data leakage, as they directly indicate attrition.

3. **Feature Engineering**: The `Tenure_Months` feature is calculated from start and exit dates, providing valuable temporal information for prediction.

4. **Model Selection**: Models are evaluated primarily on **Recall** metric, as false negatives (missing employees who will leave) are more costly than false positives in attrition prediction.

## Usage Notes

- The notebook uses a random seed (42) for reproducibility
- All models use balanced class weights or stratified sampling to handle imbalanced data
- The preprocessing pipeline includes imputation and scaling for robust model performance
- Feature importance analysis helps identify key factors driving attrition

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `employee_data.csv` is in the same directory as the notebook
2. **Import Errors**: Make sure all packages from `requirements.txt` are installed
3. **Memory Issues**: If working with larger datasets, consider reducing the number of estimators in tree-based models

## Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Additional feature engineering (e.g., interaction terms)
- SMOTE or other oversampling techniques for better handling of class imbalance
- Deployment as a web application for real-time predictions
- Model interpretation using SHAP values

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project repository or contact the maintainer.

