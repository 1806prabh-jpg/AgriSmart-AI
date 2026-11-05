# AI Crop Advisory & Farmer Decision Support System

## 1. Project Overview

The AI Crop Advisory & Farmer Decision Support System is an intelligent agriculture platform designed to help farmers make informed agricultural decisions using Machine Learning, Deep Learning, weather information and market price analysis.

The system provides four major functionalities:

1. AI-based crop recommendation.
2. Weather advisory.
3. Plant disease detection.
4. Agricultural market price analysis.

## 2. Problem Statement

Farmers often face difficulties in selecting suitable crops, monitoring changing weather conditions, identifying plant diseases and comparing agricultural market prices.

This project aims to provide a unified digital platform that supports these decisions using data-driven technologies.

## 3. Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow and Keras
- **Image Processing:** Pillow
- **Data Processing:** Pandas and NumPy
- **Visualization:** Streamlit Charts
- **Weather Data:** Open-Meteo API
- **Model Storage:** Joblib and Keras

## 4. Main Modules

### Module 1: Crop Recommendation

- Accepts Nitrogen, Phosphorus and Potassium values.
- Accepts temperature, humidity, soil pH and rainfall.
- Uses a trained Random Forest Classifier.
- Displays the recommended crop and top three model predictions.

### Module 2: Weather Advisory

- Accepts a city name.
- Retrieves weather information through Open-Meteo.
- Displays temperature, humidity, precipitation, wind speed and rain probability.
- Generates basic weather alerts using predefined thresholds.

### Module 3: Plant Disease Detection

- Accepts an uploaded plant leaf image.
- Uses a trained MobileNetV2-based deep learning model.
- Identifies a predicted crop and possible disease condition.
- Displays model confidence and top three predictions.

### Module 4: Market Price Analysis

- Displays crop prices across different mandis.
- Calculates average, highest and lowest modal prices.
- Identifies the mandi with the highest modal price in the selected dataset.
- Supports uploading a CSV file containing market prices.
- Displays a mandi-wise price comparison chart.

**Market data limitation:** The built-in market prices are illustrative demonstration data, not live market prices.

## 5. Dataset Information

### Crop Recommendation Dataset

- File: `data/Crop_recommendation.csv`
- Records: 2,200
- Features: N, P, K, temperature, humidity, pH and rainfall.
- Target: Crop label.

### Plant Disease Dataset

- Dataset: PlantVillage
- Images were obtained through TensorFlow Datasets downloads.
- Training used a selected subset of 11,527 images.
- Number of output classes: 39.

## 6. Model Performance

### Crop Recommendation

- Algorithm: Random Forest Classifier
- Number of trees: 300
- Held-out test accuracy: 99.32%

### Plant Disease Detection

- Architecture: MobileNetV2 Transfer Learning
- Image size: 160 × 160
- Training images: 9,221
- Validation images: 2,306
- Validation accuracy: 94.49%

These results are dataset-based evaluation results. They do not guarantee the same performance on real-world farm images or field conditions.

## 7. Project Structure

```text
Smart_Crop_Advisory/
│
├── app.py
│
├── data/
│   ├── Crop_recommendation.csv
│   └── plant_village/
│
├── models/
│   ├── crop_model.joblib
│   ├── crop_metrics.json
│   ├── plant_disease_model.keras
│   ├── disease_classes.json
│   └── disease_metrics.json
│
├── src/
│   ├── weather_advisory.py
│   ├── disease_detection.py
│   └── market_analysis.py
│
├── training/
│   ├── train_crop_model.py
│   └── train_disease_model.py
│
├── README.md
└── requirements.txt
```

## 8. Installation and Execution

### Step 1: Clone or download the project

Open the project folder in VS Code.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## 9. Future Scope

- Integration with verified live mandi price data.
- Addition of more regional crops and diseases.
- Multilingual support for farmers.
- Soil testing integration.
- Personalized fertilizer recommendations.
- Mobile application development.
- Improved model evaluation using independent field images.

## 10. Conclusion

The AI Crop Advisory & Farmer Decision Support System demonstrates how Machine Learning, Deep Learning, weather services and data analytics can be combined into a single agricultural decision-support platform.

The project aims to make agricultural information more accessible and support better-informed farming decisions.