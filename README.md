# 📱 Smart Phone Price Predictor

A powerful web application built with Streamlit that predicts whether a smartphone is expensive or non-expensive based on its specifications using Machine Learning.

## 🌟 Features

- **Interactive Prediction Interface**: Beautiful gradient UI with sliders and selectors for all smartphone specifications
- **Real-time Predictions**: Instant price category prediction with confidence scores
- **Batch Prediction**: Upload CSV files to predict prices for multiple devices at once
- **Visual Analytics**:
  - Probability gauges
  - Distribution charts
  - Feature importance visualization
- **Model Performance**: 93.93% accuracy with Random Forest classifier (SMOTE-balanced training)
- **Real-World Inputs**: Enter actual values (GB, mAh, inches) - automatically normalized for prediction
- **Comprehensive Features**: 30 features including processor, camera, display, battery, and connectivity specs

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or navigate to the project directory

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Make sure `best_model.joblib` is in the project directory

### Running the App

```bash
python -m streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Model Information

- **Algorithm**: Random Forest Classifier (SMOTE-Balanced Training)
- **Accuracy**: 93.93% (on validation set)
- **Macro F1 Score**: 93.93%
- **Training Data**: 1,232 samples (SMOTE-balanced: 616 per class)
- **Original Training Data**: 857 samples (imbalanced)
- **Validation Data**: 247 samples (balanced)
- **Test Data**: 153 samples (original distribution)

### Models Tested

- **Random Forest** (Selected - Best Performance: 93.93% accuracy on SMOTE-balanced data)
- XGBoost (93.52% accuracy)
- Logistic Regression (91.50% accuracy)
- SVC (Support Vector Classifier) (91.50% accuracy)

## 🎯 Usage

### Single Prediction

1. Navigate to the "🔮 Prediction" tab
2. Enter real smartphone specifications:
   - **General**: Rating (0-5 stars)
   - **Processor**: Brand (Snapdragon, MediaTek, etc.), Core Count, Clock Speed (GHz)
   - **Memory**: RAM (1-18 GB), Storage (16GB-2TB)
   - **Battery**: Capacity (2000-7000 mAh), Fast Charging (0-135W)
   - **Display**: Screen Size (4-8 inches), Resolution (pixels), Refresh Rate (60-144 Hz)
   - **Camera**: Rear/Front cameras, Megapixels (0-200 MP)
   - **Connectivity**: 4G, 5G, NFC, Dual SIM (checkboxes)
3. Click "🔮 Predict Price Category"
4. View results with confidence scores, probability gauge, and input summary

### Batch Prediction

1. Navigate to the "📊 Batch Prediction" tab
2. Upload a CSV file with smartphone specifications (normalized values)
3. Click "🚀 Run Batch Prediction"
4. View results and download predictions
5. Analyze distribution charts

### Model Info

- View detailed model metrics
- See feature importance rankings
- Understand the training pipeline

## 📁 Project Structure

```
Smart Phone Prices Prediction/
├── app.py                          # Main Streamlit application
├── best_model.joblib               # Trained Random Forest model (SMOTE-balanced)
├── requirements.txt                # Python dependencies
├── preprocessing.ipynb             # Data preprocessing notebook
├── models.ipynb                    # Model training notebook
├── balance_data.py                 # Script to balance data using various methods
├── check_balance.py                # Script to check data balance
├── train.csv                       # Original training data
├── test.csv                        # Original test data
├── train_processed.csv             # Processed training data
├── test_processed.csv              # Processed test data
├── train_balanced_smote.csv        # SMOTE-balanced training data (used for final model)
├── submission_best_model.csv       # Model predictions on test set
├── balancing_comparison.png        # Balancing methods comparison
├── smote_balancing.png             # SMOTE balancing visualization
├── Models Documentation.txt        # Model hyperparameters documentation
├── Project_Report.md               # Comprehensive technical report
├── feature_analysis.txt            # Feature importance analysis
└── Smart Phone Prices Prediction.pdf # Project documentation
```

## 🔧 Features Explained

### Input Features (30 total):

The app accepts **real-world values** and automatically normalizes them for the model:

1. **rating**: Overall device rating (0-5 stars)
2. **Core_Count**: Number of CPU cores (1-10)
3. **Clock_Speed_GHz**: Processor speed (0.5-3.5 GHz)
4. **RAM Size GB**: Amount of RAM (1-18 GB)
5. **Storage Size GB**: Storage capacity (16-2048 GB)
6. **battery_capacity**: Battery size (2000-7000 mAh)
7. **fast_charging_power**: Fast charging power (0-135W)
8. **Screen_Size**: Display size (4-8 inches)
9. **Resolution_Width**: Display width (pixels)
10. **Resolution_Height**: Display height (pixels)
11. **Refresh_Rate**: Screen refresh rate (60-144 Hz)
12. **Camera specs**: Rear/front cameras and megapixels (0-200 MP)
13. **Performance tiers**: Budget, Mid-range, High, Flagship
14. **Processor brands**: Snapdragon, MediaTek, Apple, Exynos, etc.
15. **Connectivity**: 4G, 5G, NFC, IR Blaster, Dual SIM, etc.
16. **Brand & OS**: Manufacturer and operating system

## 🎨 UI Features

- **Real-World Inputs**: Enter actual GB, mAh, inches - no confusing normalized values!
- **Auto-Normalization**: App automatically converts to model format
- **Gradient Background**: Modern purple gradient design
- **Hover Effects**: Interactive buttons with smooth transitions
- **Responsive Layout**: Works on desktop and mobile
- **Visual Feedback**: Color-coded predictions (green for budget, red for premium)
- **Charts & Gauges**: Plotly-powered visualizations
- **Input Summary**: Review all entered specifications before prediction

## 📈 Performance Metrics

| Metric                    | Score      |
| ------------------------- | ---------- |
| **Accuracy**              | **93.93%** |
| **Macro F1**              | **93.93%** |
| Precision (Non-Expensive) | 93.7%      |
| Recall (Non-Expensive)    | 94.4%      |
| F1-Score (Non-Expensive)  | 94.0%      |
| Precision (Expensive)     | 94.2%      |
| Recall (Expensive)        | 93.5%      |
| F1-Score (Expensive)      | 93.9%      |

### Test Set Results

- **Total Predictions**: 153 phones
- **Non-Expensive**: 109 (71.2%)
- **Expensive**: 44 (28.8%)

## 🛠️ Technologies Used

- **Streamlit**: Web framework for interactive UI
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning (Random Forest classifier)
- **Plotly**: Interactive visualizations and gauges
- **Joblib**: Model serialization and loading

## 📝 Notes

- **User-Friendly**: Enter real values (GB, mAh, inches) - the app handles normalization automatically
- **Binary Classification**: Model predicts 0 (Non-Expensive) or 1 (Expensive)
- **Confidence Scores**: Probability percentages for both categories
- **Batch Prediction**: Upload CSV files with normalized feature values (0-1 scale)
- **High Accuracy**: 93.93% accuracy on validation set with excellent balanced performance
- **SMOTE-Balanced Training**: Model trained on SMOTE-balanced data for fair predictions across both classes

## 📊 Data Balancing

The model was trained on SMOTE-balanced data to address class imbalance:

### Original Data:

- Class 0 (Non-Expensive): 616 samples (71.88%)
- Class 1 (Expensive): 241 samples (28.12%)
- Imbalance Ratio: 2.56

### After SMOTE Balancing:

- Class 0 (Non-Expensive): 616 samples (50%)
- Class 1 (Expensive): 616 samples (50%)
- Total Training Samples: 1,232
- Synthetic Samples Created: 375 (realistic, not random)

### Why SMOTE?

- Creates realistic synthetic samples using k-nearest neighbors
- Maintains data quality and feature relationships
- Prevents model bias toward majority class
- Industry-standard technique for handling imbalanced data

**Result**: Both classes now perform equally well (~94% F1-score for each class)!

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created for AI Project - Smart Phone Price Prediction

---

**Happy Predicting! 📱✨**
