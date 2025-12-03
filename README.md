# Smartphone Price AI Predictor 📱

An advanced machine learning application that predicts whether a smartphone is "Expensive" or "Budget-Friendly" based on its technical specifications. Built with Python, Scikit-learn, and Streamlit.

## 🚀 Features

- **AI-Powered Prediction**: Uses a Random Forest Classifier to analyze device specs.
- **Interactive UI**: Modern, dark-themed interface built with Streamlit.
- **Smart Presets**: Quick-fill buttons for Flagship, Gaming, and Budget device configurations.
- **Visual Analytics**: Radar charts to visualize device capabilities (Performance, Display, Camera, etc.).
- **Real-time Inference**: Instant price category prediction with confidence scores.

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### Running the Application
To start the web interface, simply run the provided batch file or use the command line:

**Option 1: Batch File (Windows)**
Double-click `run_app.bat`

**Option 2: Command Line**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Training the Model
If you need to retrain the model with new data:

1. Place your training data in `Data/train.csv`.
2. Run the training pipeline:
   ```bash
   python train_pipeline.py
   ```
   This will generate `model_pipeline.pkl` and `unique_values.pkl`.

## 📂 Project Structure

- `app.py`: Main Streamlit application file containing the UI and inference logic.
- `train_pipeline.py`: Script to preprocess data and train the machine learning model.
- `requirements.txt`: List of Python dependencies.
- `run_app.bat`: Shortcut to run the application on Windows.
- `Data/`: Directory for storing datasets.
- `model_pipeline.pkl`: Saved trained model pipeline.
- `unique_values.pkl`: Dictionary of unique categorical values used for UI dropdowns.

## 📦 Dependencies

- **Streamlit**: Web application framework.
- **Pandas & NumPy**: Data manipulation.
- **Scikit-learn**: Machine learning model and preprocessing.
- **Joblib**: Model serialization.
- **Plotly**: Interactive visualizations.
