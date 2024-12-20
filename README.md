# SVM Hyperparameter Optimization on the Iris Dataset

## Description
This project demonstrates the process of hyperparameter optimization for a Support Vector Machine (SVM) classifier on the Iris dataset. The script evaluates SVM performance across multiple training-testing splits, searching for the best hyperparameters using a random search approach.

## Features
- Load and preprocess the Iris dataset from `sklearn`.
- Normalize features for better model performance.
- Generate multiple (10) training-testing splits for robustness.
- Optimize SVM hyperparameters (`kernel`, `C`, and `gamma`) using random search.
- Evaluate and track model accuracy over 100 iterations for each split.
- Summarize results in a performance table.
- Plot convergence graph for the best-performing model.
- Save results to a CSV file for future reference.

## Requirements
- Python 3.7+
- Required Python Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

Install the required libraries using:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Run the Script**
   Execute the script in your terminal or IDE:
   ```bash
   python svm_optimization.py
   ```

3. **Outputs**
   - **Console Outputs:**
     - Dataset summary, class distribution, and optimal parameters for each training-testing split.
   - **Performance Summary CSV:**
     - A file named `Optimized_SVM_Performance.csv` will be generated containing:
       - Sample name
       - Best accuracy achieved
       - Optimal hyperparameters
   - **Convergence Graph:**
     - Displays the accuracy progression across 100 iterations for the best-performing sample.

## Project Workflow

### Step 1: Dataset Loading and Analysis
- Fetch the Iris dataset using `fetch_openml` from `sklearn`.
- Display dataset details including:
  - Number of samples and features.
  - Class distribution in the target variable.

### Step 2: Data Preprocessing
- Normalize features using `StandardScaler` to standardize feature values.

### Step 3: Generate Training-Testing Splits
- Create 10 training-testing splits (70%-30%) using different random seeds for diversity.

### Step 4: Hyperparameter Optimization
- Perform random search for SVM hyperparameters across 100 iterations per split.
  - Kernel types: `linear`, `poly`, `rbf`, `sigmoid`
  - Regularization parameter `C`: Uniformly sampled from [0.1, 1.0]
  - Gamma: Uniformly sampled from [0.01, 0.1]
- Record the best accuracy and parameters for each split.

### Step 5: Summarize Results
- Create a table summarizing the best accuracy and hyperparameters for all 10 splits.
- Save the results to a CSV file (`Optimized_SVM_Performance.csv`).

### Step 6: Visualize Convergence
- Plot the accuracy progression over iterations for the sample with the highest accuracy.

## Results
- The script identifies the optimal hyperparameters and best accuracy for each split.
- The convergence graph illustrates accuracy improvement over iterations.
- A summary table provides an easy-to-read format for all results.

## Future Enhancements
- Automate hyperparameter search using GridSearchCV or RandomizedSearchCV.
- Extend to other datasets or classification problems.
- Implement additional performance metrics such as precision, recall, and F1-score.

## Acknowledgments
- [Scikit-learn](https://scikit-learn.org/) for providing robust tools for machine learning.
- The Iris dataset from UCI Machine Learning Repository.
- [Matplotlib](https://matplotlib.org/) for visualization.

