import gradio as gr
import torch
import torch.nn as nn
import pickle
import numpy as np

# Define model architecture
class SalaryPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Load model and preprocessors
model = SalaryPredictor()
model.load_state_dict(torch.load('salary_model.pth', map_location='cpu'))
model.eval()

scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

def predict_salary(age, gender, education, job_title, years_experience):
    try:
        # Encode categorical variables
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        education_encoded = label_encoders['education'].transform([education])[0]
        job_encoded = label_encoders['job'].transform([job_title])[0]
        
        # Create feature array
        features = np.array([[age, gender_encoded, education_encoded, job_encoded, years_experience]])
        
        # Scale features
        features_scaled = scaler_X.transform(features)
        
        # Predict
        with torch.no_grad():
            prediction_scaled = model(torch.FloatTensor(features_scaled))
        
        # Convert back to salary
        salary = scaler_y.inverse_transform(prediction_scaled.numpy())[0][0]
        
        return f"Predicted Salary: ${salary:,.2f}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="Age", value=30, minimum=18, maximum=70),
        gr.Dropdown(["Male", "Female", "Other"], label="Gender", value="Female"),
        gr.Dropdown(["High School", "Bachelor's", "Master's", "PhD"], label="Education Level", value="Bachelor's"),
        gr.Dropdown(["Data Analyst", "Data Scientist", "Software Engineer", "Product Manager", "Other"], label="Job Title", value="Data Analyst"),
        gr.Number(label="Years of Experience", value=5, minimum=0, maximum=50)
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="💼 Salary Prediction Service",
    description="Enter employee information to predict salary using a PyTorch neural network model.",
    examples=[
        [30, "Female", "Bachelor's", "Data Analyst", 5],
        [35, "Male", "Master's", "Data Scientist", 8],
        [28, "Female", "Bachelor's", "Software Engineer", 3]
    ]
)

if __name__ == "__main__":
    demo.launch()