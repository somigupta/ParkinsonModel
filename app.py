
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # if you want to load scaler later
from fastapi.middleware.cors import CORSMiddleware

# =========================
# 1. Define Model (same as training)
# =========================
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        
        self._to_linear = None
        self._get_conv_output(input_size)

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# =========================
# 2. Initialize FastAPI
# =========================
app = FastAPI(title="Parkinson's Detection API")
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# =========================
# 3. Load Model
# =========================
INPUT_SIZE = 22  # adjust if your dataset differs

model = CNNModel(INPUT_SIZE)
model.load_state_dict(torch.load("parkinsons_cnn_model.pkl", map_location=torch.device('cpu')))
model.eval()

# =========================
# 4. Input Schema
# =========================
class PatientData(BaseModel):
    features: list  # list of 22 float values

# =========================
# 5. Prediction Endpoint
# =========================
@app.post("/predict")
def predict(data: PatientData):
    try:
        features = np.array(data.features, dtype=np.float32)
        features = torch.tensor(features).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(features).item()

        prediction = 1 if output > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": float(output),
            "label": "Parkinson's" if prediction == 1 else "Healthy"
        }

    except Exception as e:
        return {"error": str(e)}

# =========================
# 6. Root Endpoint
# =========================
@app.get("/")
def home():
    return {"message": "Parkinson's CNN API is running"}
