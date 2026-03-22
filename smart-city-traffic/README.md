#  Smart City Traffic Pattern Forecasting

> Machine Learning system for predicting traffic patterns across 4 city junctions

---

##  Project Overview

A government-commissioned smart city initiative to forecast traffic volumes at four key city junctions. The system accounts for holidays, weekends, seasonal patterns, and time-of-day variations to enable proactive traffic management and infrastructure planning.

---

##  Project Structure

```
smart-city-traffic/
│
├── data/
│   ├── raw/traffic_data.csv          ← Original dataset
│   └── processed/cleaned_data.csv   ← Preprocessed data
│
├── notebooks/
│   ├── 01_eda.ipynb                  ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        ← Data Cleaning & Feature Engineering
│   └── 03_model_training.ipynb       ← Model Building & Evaluation
│
├── src/
│   ├── data_preprocessing.py         ← Data cleaning pipeline
│   ├── feature_engineering.py        ← Feature creation
│   ├── train_model.py                ← Model training
│   ├── evaluate_model.py             ← Model evaluation metrics
│   └── predict.py                    ← Prediction interface
│
├── models/
│   └── traffic_model.pkl             ← Saved trained model
│
├── app/
│   └── streamlit_app.py              ← Interactive dashboard
│
├── results/
│   ├── graphs/                       ← Saved visualizations
│   └── metrics.txt                   ← Model performance results
│
├── main.py                           ← Run full pipeline
├── requirements.txt
└── README.md
```

---

##  Setup & Installation

```bash
# Clone / navigate to project
cd smart-city-traffic

# Install dependencies
pip install -r requirements.txt

# Place dataset
# Download from: https://drive.google.com/file/d/1y61cDyuO9Zrp1fSchWcAmCxk0B6SMx7X/
# Save as: data/raw/traffic_data.csv
```

---

##  Running the Project

### Option 1: Full Pipeline (Training + Evaluation)
```bash
python main.py
```

### Option 2: Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Option 3: Step-by-step Notebooks
```bash
jupyter notebook notebooks/
```

---

##  Dataset

| Column     | Description                          |
|------------|--------------------------------------|
| DateTime   | Timestamp of observation             |
| Junction   | Junction ID (1–4)                    |
| Vehicles   | Number of vehicles counted           |
| ID         | Unique record identifier             |

---

##  Models Used

| Model              | Purpose                        |
|--------------------|--------------------------------|
| Linear Regression  | Baseline traffic prediction    |
| Random Forest      | Advanced ensemble prediction   |

### Evaluation Metrics
- **RMSE** – Root Mean Square Error  
- **MAE** – Mean Absolute Error  
- **R²** – Coefficient of Determination  

---

##  Key Insights

- Peak traffic typically occurs between **7–9 AM** and **5–7 PM**
- Weekend traffic is **20–30% lower** than weekday traffic
- Junction 1 consistently shows **highest vehicle volumes**
- Holiday periods show **distinct traffic signatures**

---

##  Real-World Impact

 Enables proactive traffic signal timing  
 Supports infrastructure planning decisions  
 Reduces average commute time  
 Provides data-driven policy inputs  

---

##  Future Enhancements

- Real-time data ingestion via IoT sensors
- LSTM/Deep Learning models for improved accuracy
- Mobile app for citizen alerts
- Smart signal automation integration
