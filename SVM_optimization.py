
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Step 1: Importing and analyzing the dataset
print("Loading dataset...")
uci_dataset = fetch_openml("iris", version=1)  # Use a multi-class dataset from UCI repository
data_frame = pd.DataFrame(uci_dataset.data, columns=uci_dataset.feature_names)
data_frame['target'] = uci_dataset.target

# Display basic details about the dataset
print("\nDataset Summary:")
print(data_frame.info())
print("\nFirst Few Rows:")
print(data_frame.head())
print("\nClass Distribution in the Target Variable:")
print(data_frame['target'].value_counts())

# Step 2: Preprocessing the data
features = data_frame.drop(columns=['target'])
target = data_frame['target']

# Normalize the feature data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Step 3: Generate multiple training-testing samples (10 samples, 70-30 split)
data_splits = []
for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, target, test_size=0.3, random_state=seed
    )
    data_splits.append((X_train, X_test, y_train, y_test))

# Step 4: Optimize SVM for each sample
optimization_results = []

for sample_index, (X_train, X_test, y_train, y_test) in enumerate(data_splits):
    best_accuracy = 0
    optimal_parameters = None
    accuracy_progress = []

    # Perform optimization over 100 iterations
    for iteration in range(100):
        # Randomly choose SVM hyperparameters
        chosen_kernel = np.random.choice(['linear', 'poly', 'rbf', 'sigmoid'])
        chosen_c = np.random.uniform(0.1, 1.0)  # C is the regularization parameter
        chosen_gamma = np.random.uniform(0.01, 0.1)

        # Train the SVM model
        svm_model = SVC(kernel=chosen_kernel, C=chosen_c, gamma=chosen_gamma, random_state=42)
        svm_model.fit(X_train, y_train)
        predictions = svm_model.predict(X_test)

        # Calculate accuracy
        current_accuracy = accuracy_score(y_test, predictions)
        accuracy_progress.append(current_accuracy)

        # Check for the best performance
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            optimal_parameters = (chosen_kernel, chosen_c, chosen_gamma)

    # Store the results for the current sample
    optimization_results.append({
        "Sample": f"Sample-{sample_index + 1}",
        "Best Accuracy": best_accuracy,
        "Optimal Parameters": optimal_parameters,
        "Accuracy Progress": accuracy_progress
    })

    print(f"\nSample {sample_index + 1} - Best Accuracy: {best_accuracy:.4f}")
    print(f"Optimal Parameters: Kernel={optimal_parameters[0]}, C={optimal_parameters[1]:.4f}, Gamma={optimal_parameters[2]:.4f}")

# Step 5: Create a performance summary table
summary_df = pd.DataFrame({
    "Sample": [result["Sample"] for result in optimization_results],
    "Best Accuracy": [result["Best Accuracy"] for result in optimization_results],
    "Optimal Parameters": [result["Optimal Parameters"] for result in optimization_results]
})
print("\nPerformance Summary Table:")
print(summary_df)

# Step 6: Plot convergence graph for the sample with the highest accuracy
top_sample = max(optimization_results, key=lambda res: res["Best Accuracy"])
plt.figure(figsize=(8, 6))
plt.plot(top_sample["Accuracy Progress"], label="Accuracy Over Iterations", color="blue", linewidth=2)
plt.title("Convergence Graph for the Best SVM Model", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Step 7: Save results to CSV file
output_file = "Optimized_SVM_Performance.csv"
summary_df.to_csv(output_file, index=False)
print(f"\nResults have been saved to '{output_file}'. Consider uploading this file to your GitHub repository.")