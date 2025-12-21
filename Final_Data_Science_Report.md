# Final Data Science Report: Smartphone Price Classification

## 1. Preprocessing Techniques
Before modeling, the raw dataset underwent rigorous preprocessing to ensure data quality and model compatibility.

### A. Data Cleaning & Handling
*   **Duplicate Removal**: Using `df.drop_duplicates()`, we identified and removed identical rows to prevent data leakage and bias.
*   **Missing Value Check**: We verified that there were no missing values (`NaN`) in the critical columns, ensuring complete data for training.
*   **Column Dropping**: Irrelevant identifiers or redundant columns were removed to reduce noise.

### B. Feature Encoding
Machine learning models require numerical input, so categorical variables were transformed:
*   **Binary Encoding**: Features with two states (e.g., `4G`, `5G`, `Dual_Sim`, `NFC`) were mapped to binary values (`0` for No, `1` for Yes).
*   **Label Encoding**: Cardinal variables like `Brand`, `OS`, and `Processor_Brand` were encoded into numerical labels (e.g., `brand_encoded_label`) to preserve their categorical distinction without creating excessive dimensionality.
*   **Target Encoding**: The target variable `price` was encoded as:
    *   `0`: Non-Expensive
    *   `1`: Expensive

### C. Data Scaling & Balancing
*   **Balancing (SMOTE)**: The dataset was imbalanced (fewer expensive phones). We applied **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class, ensuring the model doesn't become biased toward the majority class.
*   **Scaling**: For distance-based algorithms (SVM, Logistic Regression), we used `StandardScaler` within a Pipeline to normalize features (Mean=0, Std=1).

---

## 2. Data Analysis & Visualization
We analyzed 1,232 samples to understand feature relationships.

### Key Correlations (Insights)
*   **Speed & Power Drive Price**: The strongest positive correlations with Price are **Processor Clock Speed** (`0.58`) and **RAM Size** (`0.54`).
*   **Display Quality**: **Resolution** (`0.43`) and **Refresh Rate** (`0.42`) are also strong indicators of premium devices.
*   **The Battery Paradox**: **Battery Capacity** has a weak **negative** correlation (`-0.19`) with price. Premium phones often ignore massive batteries in favor of faster charging (`0.39` correlation) and slim aesthetics, whereas budget phones market large batteries as a primary feature.

---

## 3. Modeling & Hyperparameters
We trained and evaluated four distinct algorithms. The **Random Forest** model performed best.

### A. Random Forest (Best Model)
*   **Accuracy**: **95.55%**
*   **Hyperparameters**:
    *   `n_estimators=200`: Increased tree count for stability.
    *   `max_depth=9`: Controlled depth to prevent overfitting.
    *   `n_jobs=-1`: Parallel processing.

### B. Support Vector Classifier (SVC)
*   **Accuracy**: 93.93%
*   **Hyperparameters**:
    *   `kernel='poly'`: Polynomial kernel handled non-linear boundaries well.
    *   `C=1.0`: Standard regularization.
    *   `class_weight='balanced'`: To handle any remaining imbalance.

### C. XGBoost
*   **Accuracy**: 93.52%
*   **Hyperparameters**:
    *   `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`.

### D. Logistic Regression
*   **Accuracy**: 91.50%
*   **Hyperparameters**:
    *   `C=100`: Low regularization (high trust in data).
    *   `solver='lbfgs'`: Optimized for convergence.

---

## 4. Enhancement Techniques
To maximize performance, we employed:
1.  **GridSearchCV**: Systematically searched for optimal hyperparameters (e.g., finding that `max_depth=9` was better than `15` for RF).
2.  **SMOTE**: Fixed class imbalance, significantly improving recall for the "Expensive" class.
3.  **Pipelines**: Encapsulated Scaling + Modeling into single objects to prevent data leakage during cross-validation.

---

## 5. Conclusion
This project successfully built a robust pricing classifier with **95.55% accuracy**. The analysis proves that **computation power (CPU/RAM)** is the primary driver of smartphone cost, outweighing features like battery life. The final **Random Forest** model is deployed in a streamlined web application, ready for real-world usage.
