# Deep-learning
Detailed Step-by-Step Description of the Project
Implementing and Optimizing a Custom Gradient Boosting Machine

---

Step 1: Problem Definition and Objective

The objective of this project is to design and implement a Gradient Boosting Machine (GBM) for a regression task from scratch using only NumPy or SciPy. The project aims to replicate the core learning principles behind popular implementations such as XGBoost and LightGBM while avoiding the use of scikit-learn’s estimator classes. The focus is on understanding the internal working of gradient boosting, including loss minimization, residual learning, decision tree construction, and performance optimization. The project also includes a comparative evaluation against scikit-learn’s GradientBoostingRegressor to analyze trade-offs between custom and library-based implementations.

---

Step 2: Dataset Selection and Preparation

A standard built-in regression dataset such as the California Housing dataset is used to ensure consistency and reproducibility. The dataset consists of numerical features and a continuous target variable, making it suitable for regression and tree-based learning. Since the dataset is already cleaned and contains no missing values or categorical variables, no additional data preprocessing steps such as normalization, standardization, or encoding are applied. The dataset is split into training and testing sets using the same split configuration for both the custom GBM and the scikit-learn model to ensure a fair comparison.

---

Step 3: Initial Model Prediction (Task 1)

Gradient boosting begins with a simple baseline model. In this project, the initial prediction is calculated as the mean of the target values in the training data. This constant prediction minimizes the mean squared error loss for a constant model. The initial prediction is stored and used as the starting point for the boosting process. All subsequent models attempt to correct the errors made by this initial prediction.

---

Step 4: Loss Function Definition and Pseudo-Residual Computation (Task 2)

The mean squared error (MSE) loss function is selected for this regression task. The loss function measures the average squared difference between the true target values and the predicted values. Pseudo-residuals are computed as the negative gradient of the loss function with respect to the predictions. For MSE loss, the pseudo-residuals simplify to the difference between the actual target values and the current predictions. These residuals represent the information that the next base learner must model.

---

Step 5: Decision Tree Structure Design (Task 3)

A custom decision tree regressor is implemented to serve as the base learner for the gradient boosting framework. Each tree consists of internal nodes and leaf nodes. Internal nodes store the selected feature index and split threshold, while leaf nodes store a prediction value. The depth of the tree and the minimum number of samples required at a node are controlled using hyperparameters to prevent excessive growth and overfitting.

---

Step 6: Split Criterion and Optimal Split Selection (Task 3)

At each internal node, the algorithm evaluates potential splits using a variance reduction or mean squared error criterion. For each feature, the data is sorted based on feature values, and all possible split points are evaluated. To improve efficiency, vectorised computations using cumulative sums are employed. This allows the mean squared error for all split points to be computed simultaneously without nested loops. The feature and threshold that result in the lowest combined error for the left and right child nodes are selected as the optimal split.

---

Step 7: Optimal Leaf Value Calculation (Task 3)

When a node satisfies a stopping condition, such as reaching the maximum tree depth or having too few samples, it becomes a leaf node. The optimal value assigned to a leaf is calculated as the mean of the residuals in that node. This value minimizes the mean squared error loss for that subset of data. The leaf value represents the correction applied to the current prediction during boosting.

---

Step 8: Recursive Tree Growth (Task 3)

Using the selected optimal split, the dataset is divided into left and right subsets. The tree-building process is applied recursively to each subset until stopping conditions are met. This results in a shallow decision tree that captures local patterns in the residuals. The recursion ensures that each tree incrementally improves the overall model.

---

Step 9: Boosting Iteration and Model Update (Task 1 and Task 2)

The gradient boosting process consists of multiple boosting iterations. In each iteration, a new decision tree is trained to predict the pseudo-residuals from the previous stage. The predictions from the tree are scaled by a learning rate and added to the current model predictions. This gradual update strategy prevents overfitting and improves generalization. Each trained tree is stored for use during prediction.

---

Step 10: Model Prediction

To generate predictions on unseen data, the model starts with the initial mean prediction and sequentially adds the scaled predictions from all trained trees. This cumulative prediction represents the final output of the gradient boosting model.

---

Step 11: Performance Evaluation (Task 4)

The performance of the custom GBM is evaluated using Root Mean Squared Error (RMSE) on the test dataset. Training time is also measured to assess computational efficiency. The same evaluation metrics are applied to scikit-learn’s GradientBoostingRegressor trained on the same dataset with comparable hyperparameters. This ensures a fair and consistent comparison.

---

Step 12: Optimization and Performance Analysis (Task 4)

The optimized custom implementation is compared against a naive version to demonstrate the impact of vectorised operations. The use of cumulative sums and NumPy-based computations significantly reduces training time. However, the scikit-learn implementation remains faster and more accurate due to its highly optimized C and C++ backend. The analysis highlights the trade-off between algorithmic transparency and computational efficiency.

---

Step 13: Final Comparison and Conclusion

The project concludes with a detailed comparison of the custom and scikit-learn implementations in terms of RMSE, training time, and implementation complexity. While the scikit-learn model is superior in speed and accuracy, the custom GBM provides a deep understanding of gradient boosting mechanics and demonstrates how performance can be improved through algorithmic optimization. The project successfully meets its objective of building and optimizing a Gradient Boosting Machine . 
