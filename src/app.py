#  python -m streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Smart Phone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .prediction-box {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    h1, h2, h3 {
        color: white;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: white !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Normalization functions (based on typical smartphone ranges)
def normalize_rating(val):
    """Rating is already 0-5, normalize to 0-1"""
    return val / 5.0

def normalize_clock_speed(val):
    """Clock speed typically 0.5 - 3.5 GHz"""
    return (val - 0.5) / 3.0

def normalize_ram(val):
    """RAM typically 1-18 GB"""
    return (val - 1) / 17.0

def normalize_storage(val):
    """Storage typically 16-2048 GB"""
    return (val - 16) / 2032.0

def normalize_battery(val):
    """Battery typically 2000-7000 mAh"""
    return (val - 2000) / 5000.0

def normalize_fast_charging(val):
    """ Fast charging 0-135W"""
    return val / 135.0

def normalize_screen_size(val):
    """Screen size typically 4-8 inches"""
    return (val - 4) / 4.0

def normalize_resolution_dim(val, max_val=4000):
    """Resolution dimensions"""
    return val / max_val

def normalize_refresh_rate(val):
    """Refresh rate typically 60-144 Hz"""
    return (val - 60) / 84.0

def normalize_camera_mp(val):
    """Camera MP typically 0-200 MP"""
    return val / 200.0

# Title and description
st.title("📱 Smart Phone Price Predictor")
st.markdown("""
<div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='color: white;'>Predict if a smartphone is Expensive or Non-Expensive</h3>
    <p style='color: white;'>Enter real smartphone specifications below to get an instant price category prediction.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This application uses a Random Forest model trained on SMOTE-balanced data to predict smartphone price categories.

**Enter real values** - the app automatically normalizes them for the model.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.success("Accuracy: 95.55%")
st.sidebar.info("Macro F1 Score: 95.55%")
st.sidebar.info("Training: SMOTE-Balanced Data")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Batch Prediction", "📈 Model Info"])

with tab1:
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 General")
        rating = st.slider("Rating (Stars)", 0.0, 5.0, 4.0, 0.1)
        
        st.markdown("### 🔧 Processor")
        core_count = st.selectbox("Core Count", [1, 2, 4, 6, 8, 10], index=4)
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.5, 2.8, 0.1)
        performance_tier = st.selectbox("Performance Tier", 
                                       ["Budget (1)", "Mid-range (2)", "High (3)", "Flagship (4)"], 
                                       index=3)
        processor_brand = st.selectbox("Processor Brand", 
                                      ["Snapdragon", "MediaTek", "Apple", "Exynos", "Kirin", "Tensor", "Unisoc", "Other"],
                                      index=0)
        
        st.markdown("### 💾 Memory")
        ram_gb = st.selectbox("RAM (GB)", [1, 2, 3, 4, 6, 8, 12, 16, 18], index=5)
        storage_gb = st.selectbox("Storage (GB)", [16, 32, 64, 128, 256, 512, 1000, 2000], index=6)
        ram_tier = st.selectbox("RAM Tier", ["Low (1)", "Medium (2)", "High (3)", "Ultra (4)"], index=0)
    
    with col2:
        st.markdown("### 🔋 Battery")
        battery_capacity = st.slider("Battery Capacity (mAh)", 2000, 7000, 4500, 100)
        fast_charging_power = st.slider("Fast Charging (W)", 0, 135, 65, 5)
        
        st.markdown("### 📺 Display")
        screen_size = st.slider("Screen Size (inches)", 4.0, 8.0, 6.5, 0.1)
        resolution_width = st.number_input("Resolution Width (px)", 0, 4000, 1080, 10)
        resolution_height = st.number_input("Resolution Height (px)", 0, 4000, 2400, 10)
        refresh_rate = st.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144], index=2)
        notch_type = st.selectbox("Notch Type", 
                                  ["No Notch", "Waterdrop", "Punch Hole", "Wide Notch", "Dynamic Island", "Under Display", "Pop-up"],
                                  index=2)
        
    with col3:
        st.markdown("### 📷 Camera")
        primary_rear_camera_mp = st.slider("Primary Rear Camera (MP)", 0, 200, 50, 1)
        num_rear_cameras = st.selectbox("Number of Rear Cameras", [0, 1, 2, 3, 4, 5], index=1)
        primary_front_camera_mp = st.slider("Primary Front Camera (MP)", 0, 100, 32, 1)
        num_front_cameras = st.selectbox("Number of Front Cameras", [0, 1, 2, 3], index=1)
        
        st.markdown("### 🔌 Connectivity")
        has_4g = st.checkbox("4G Support", value=True)
        has_5g = st.checkbox("5G Support", value=True)
        dual_sim = st.checkbox("Dual SIM", value=True)
        vo5g = st.checkbox("Vo5G Support", value=False)
        nfc = st.checkbox("NFC", value=True)
        ir_blaster = st.checkbox("IR Blaster", value=False)
        memory_card = st.checkbox("Memory Card Support", value=True)
        
    # Additional features in expandable section
    with st.expander("🎨 Additional Details"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            os_name = st.selectbox("OS", ["Android", "iOS", "HarmonyOS", "ColorOS", "MIUI", "OneUI", "OxygenOS", "Other"], index=0)
            
            # Brand mapping (from label encoding in preprocessing)
            brand_options = {
                "Samsung": 17,
                "Xiaomi": 43,
                "Vivo": 38,
                "Realme": 29,
                "OPPO": 14,
                "Apple": 1,
                "Motorola": 12,
                "OnePlus": 16,
                "Tecno": 34,
                "iQOO": 24,
                "Infinix": 23,
                "Poco": 27,
                "Nokia": 13,
                "Oppo": 15,
                "Huawei": 22,
                "Google": 19,
                "Honor": 21,
                "Asus": 2,
                "Sony": 32,
                "Nothing": 28,
                "Other": 0
            }
            brand_name = st.selectbox("Brand", list(brand_options.keys()), index=0)
            brand_encoded = brand_options[brand_name]
            
        with col_b:
            os_version = st.number_input("OS Version", 0, 30, 13, 1,
                                        help="OS version number (e.g., Android 13, iOS 16)")
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Price Category", use_container_width=True):
        if model is not None:
            # Map selections to encoded values
            perf_tier_map = {"Budget (1)": 1, "Mid-range (2)": 2, "High (3)": 3, "Flagship (4)": 4}
            proc_brand_map = {"Snapdragon": 1, "MediaTek": 2, "Apple": 3, "Exynos": 4, "Kirin": 5, "Tensor": 6, "Unisoc": 7, "Other": 8}
            ram_tier_map = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3, "Ultra (4)": 4}
            notch_map = {"No Notch": 0, "Waterdrop": 1, "Punch Hole": 2, "Wide Notch": 3, "Dynamic Island": 4, "Under Display": 5, "Pop-up": 6}
            os_map = {"Android": 1, "iOS": 2, "HarmonyOS": 3, "ColorOS": 4, "MIUI": 5, "OneUI": 6, "OxygenOS": 7, "Other": 0}
            
            # WARNING: Data in training file is ALREADY NORMALIZED (0-1 range)
            # We need to normalize user input to match!
            
            # Normalize real values to 0-1 range (matching training data)
            rating_norm = normalize_rating(rating)
            clock_speed_norm = normalize_clock_speed(clock_speed)
            ram_norm = normalize_ram(ram_gb)
            storage_norm = normalize_storage(storage_gb)
            battery_norm = normalize_battery(battery_capacity)
            fast_charging_norm = normalize_fast_charging(fast_charging_power)
            screen_norm = normalize_screen_size(screen_size)
            res_width_norm = normalize_resolution_dim(resolution_width)
            res_height_norm = normalize_resolution_dim(resolution_height)
            refresh_norm = normalize_refresh_rate(refresh_rate)
            rear_cam_norm = normalize_camera_mp(primary_rear_camera_mp)
            front_cam_norm = normalize_camera_mp(primary_front_camera_mp)
            
            # Normalize storage_gb (16-2000 GB range)
            storage_gb_norm = (storage_gb - 16) / (2000 - 16)
            
            # Prepare input data - MUST match exact feature order from model
            features = pd.DataFrame({
                'rating': [rating_norm],
                'Core_Count': [core_count / 10.0],  # Normalize to 0-1
                'Clock_Speed_GHz': [clock_speed_norm],
                'RAM Size GB': [ram_norm],
                'Storage Size GB': [storage_norm],
                'battery_capacity': [battery_norm],
                'fast_charging_power': [fast_charging_norm],
                'Screen_Size': [screen_norm],
                'Resolution_Width': [res_width_norm],
                'Resolution_Height': [res_height_norm],
                'Refresh_Rate': [refresh_norm],
                'primary_rear_camera_mp': [rear_cam_norm],
                'num_rear_cameras': [num_rear_cameras / 5.0],  # Normalize to 0-1
                'primary_front_camera_mp': [front_cam_norm],
                'num_front_cameras': [num_front_cameras / 3.0],  # Normalize to 0-1
                'storage_gb': [float(storage_gb)],  # Keep as raw value (matches training data)
                'Performance_Tier_Encoded': [perf_tier_map[performance_tier]],
                'Processor_Brand_Encoded': [proc_brand_map[processor_brand]],
                'RAM_Tier_Encoded': [ram_tier_map[ram_tier]],
                'Notch_Type_Encoded': [notch_map[notch_type]],
                '4G_Encoded': [1 if has_4g else 0],
                'Dual_Sim_Encoded': [1 if dual_sim else 0],
                '5G_Encoded': [1 if has_5g else 0],
                'Vo5G_Encoded': [1 if vo5g else 0],
                'NFC_Encoded': [1 if nfc else 0],
                'IR_Blaster_Encoded': [1 if ir_blaster else 0],
                'memory_card_support_Encoded': [1 if memory_card else 0],
                'os_name_Encoded': [os_map[os_name]],
                'brand_encoded_label': [brand_encoded],
                'os_version_label': [os_version]
            })
            
            try:
                # Make prediction
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]
                
                # Display result
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    if prediction == 0:
                        st.success("### 💰 Non-Expensive")
                        st.markdown("""
                        <div class='prediction-box'>
                            <h2 style='color: #28a745;'>✅ Budget-Friendly Device</h2>
                            <p style='color: #666;'>This smartphone is predicted to be in the <strong>non-expensive</strong> category.<br>
                            Great value for money!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("### 💎 Expensive")
                        st.markdown("""
                        <div class='prediction-box'>
                            <h2 style='color: #dc3545;'>💎 Premium Device</h2>
                            <p style='color: #666;'>This smartphone is predicted to be in the <strong>expensive</strong> category.<br>
                            Top-tier flagship device!</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_result2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction_proba[1] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Expensive Probability (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show probabilities
                st.markdown("### 📊 Confidence Scores")
                col_prob1, col_prob2 = st.columns(2)
                
                with col_prob1:
                    st.metric("Non-Expensive", f"{prediction_proba[0]*100:.2f}%", 
                             delta="Budget-Friendly" if prediction == 0 else None)
                
                with col_prob2:
                    st.metric("Expensive", f"{prediction_proba[1]*100:.2f}%",
                             delta="Premium" if prediction == 1 else None)
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.write(f"**📱 Device Specs:**")
                        st.write(f"- Rating: {rating:.1f}★")
                        st.write(f"- Cores: {core_count}")
                        st.write(f"- Clock: {clock_speed} GHz")
                        st.write(f"- RAM: {ram_gb} GB")
                        st.write(f"- Storage: {storage_gb} GB")
                    with col_s2:
                        st.write(f"**🔋 Battery & Display:**")
                        st.write(f"- Battery: {battery_capacity} mAh")
                        st.write(f"- Fast Charge: {fast_charging_power}W")
                        st.write(f"- Screen: {screen_size}\"")
                        st.write(f"- Resolution: {resolution_width}x{resolution_height}")
                        st.write(f"- Refresh: {refresh_rate} Hz")
                    with col_s3:
                        st.write(f"**📷 Cameras:**")
                        st.write(f"- Rear: {primary_rear_camera_mp}MP ({num_rear_cameras} cameras)")
                        st.write(f"- Front: {primary_front_camera_mp}MP ({num_front_cameras} camera)")
                        st.write(f"**🔌 Connectivity:**")
                        st.write(f"- 4G: {'✅' if has_4g else '❌'} | 5G: {'✅' if has_5g else '❌'}")
                        st.write(f"- NFC: {'✅' if nfc else '❌'} | Dual SIM: {'✅' if dual_sim else '❌'}")
                        
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Please check your inputs and try again.")

with tab2:
    st.markdown("### 📊 Batch Prediction")
    st.info("Upload a CSV file with smartphone specifications (normalized values expected).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("#### Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Prediction"):
                if model is not None:
                    # Define expected features in exact order
                    expected_features = [
                        'rating', 'Core_Count', 'Clock_Speed_GHz', 'RAM Size GB', 'Storage Size GB',
                        'battery_capacity', 'fast_charging_power', 'Screen_Size', 'Resolution_Width',
                        'Resolution_Height', 'Refresh_Rate', 'primary_rear_camera_mp', 'num_rear_cameras',
                        'primary_front_camera_mp', 'num_front_cameras', 'storage_gb', 'Performance_Tier_Encoded',
                        'Processor_Brand_Encoded', 'RAM_Tier_Encoded', 'Notch_Type_Encoded', '4G_Encoded',
                        'Dual_Sim_Encoded', '5G_Encoded', 'Vo5G_Encoded', 'NFC_Encoded', 'IR_Blaster_Encoded',
                        'memory_card_support_Encoded', 'os_name_Encoded', 'brand_encoded_label', 'os_version_label'
                    ]
                    
                    # Filter dataframe to keep only expected features
                    # Use reindex to ignore extra columns and fill missing ones with 0 (or handle as error if preferred)
                    # Here we assume missing columns might be an issue, but the error was about EXTRA columns.
                    # So we just select the columns that exist in both.
                    
                    # Better approach: Select only expected features. 
                    # If any are missing, it will raise a KeyError which is good (user needs to provide them).
                    X_batch = df[expected_features]
                    
                    predictions = model.predict(X_batch)
                    predictions_proba = model.predict_proba(X_batch)
                    
                    df['Prediction'] = ['Non-Expensive' if p == 0 else 'Expensive' for p in predictions]
                    df['Expensive_Probability'] = predictions_proba[:, 1]
                    
                    st.write("#### Prediction Results:")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Visualizations
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        fig = px.pie(df, names='Prediction', title='Price Category Distribution',
                                    color_discrete_map={'Non-Expensive': 'lightgreen', 'Expensive': 'lightcoral'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_v2:
                        fig = px.histogram(df, x='Expensive_Probability', nbins=20,
                                         title='Probability Distribution',
                                         color_discrete_sequence=['#667eea'])
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.markdown("### 📈 Model Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        #### 🎯 Model Details
        - **Algorithm**: Random Forest Classifier
        - **Accuracy**: 95.55%
        - **Macro F1 Score**: 95.55%
        - **Training Method**: SMOTE-Balanced Data
        - **Training Samples**: 1,232 (616 per class)
        - **Validation Samples**: 247 (balanced)
        
        #### 🏆 Top Predictors (Most Important Features)
        1. **Processor Speed** (Clock Speed GHz) 🚀
        2. **RAM Size** (GB) 💾
        3. **Resolution** (Width/Height) 📺
        4. **Refresh Rate** (Hz) 🔄
        5. **Fast Charging** (W) ⚡
        
        #### 💡 Key Data Insights
        - **Speed is King**: The fastest processors almost always belong to the 'Expensive' category.
        - **The Battery Paradox**: Expensive phones often have *smaller* batteries than budget phones! They prioritize slimness and efficiency over raw capacity.
        - **Display Matters**: High refresh rates (120Hz+) are a strong indicator of a premium device.
        """)
    
    with col_info2:
        st.markdown("""
        #### 🔄 Model Training Pipeline
        1. Data Preprocessing
        2. Feature Engineering & Normalization
        3. Label Encoding
        4. Train-Test Split (80/20)
        5. Model Training (4 algorithms tested)
        6. Hyperparameter Tuning
        7. Model Evaluation
        8. Best Model Selection (Random Forest)
        
        #### ✨ App Features
        - **Real Values**: Enter actual specs (GB, mAh, etc.)
        - **Auto Normalization**: Converts to model format
        - **Instant Prediction**: Real-time results
        - **Batch Processing**: Upload CSV files
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>📱 Smart Phone Price Predictor | Built with Streamlit & Random Forest</p>
    <p>SMOTE-Balanced Training Data | Accuracy: 95.55% | F1-Score: 95.55%</p>
</div>
""", unsafe_allow_html=True)