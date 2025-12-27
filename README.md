# Salary-Prediction-Microservice

A PyTorch-based ML application that predicts employee salaries. Built to learn end-to-end model deployment from training to production service. 

**Note: Learning project with known limitations.**

🔗 **Live Demo**: [https://lisatran183-salary-prediction.hf.space/](https://lisatran183-salary-prediction.hf.space/)

## What This Demonstrates

✅ Data preprocessing (missing values, encoding, scaling)  
✅ PyTorch neural network training  
✅ Cloud deployment with Gradio UI and REST API  
✅ Critical evaluation of model limitations

## Technology Stack

PyTorch • Hugging Face Spaces • Gradio • scikit-learn • Python

## Model Performance & Limitations

**Metrics:** R² = 0.057 | MAE = $44,511 | RMSE = $51,853

**Why it's not production-ready:**
- Limited features (only age, gender, education, job title, experience)
- Small dataset (6,698 records) with quality issues
- Missing critical variables (location, company size, skills, industry)

**Key Learning:** Deployment skills ≠ model quality. Dataset quality and feature engineering matter more than model complexity.

## Technical Implementation

**Preprocessing solved:**
- Handled missing values with `dropna()`
- LabelEncoder for categorical variables
- StandardScaler for feature normalization
- Grouped 191 job titles → 21 categories
- Proper train/test split pipeline

## Local Development
```bash
git clone https://github.com/lisatran183/salary-prediction.git
pip install -r requirements.txt
python app.py
```

## Why Share This?

Real learning means understanding what didn't work. This project shows I can deploy ML models, understand the gap between technical functionality and business value, and critically evaluate my own work.

## Author

**Lisa Tran** | Master's in Analytics, Northeastern University | 4.0 GPA | Graduating March 2026

📧 thaotranthuynhu@gmail.com | 🔗 [LinkedIn](https://www.linkedin.com/in/lisa-tran-analytics/) | 💻 [GitHub](https://github.com/lisatran183)

---

**Acknowledgments:** [Dataset](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data) • [Tutorial](https://machinelearningmastery.com/building-a-regression-model-in-pytorch/)
