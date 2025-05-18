import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define symptom order (must be consistent)
symptom_order = ['fever', 'cough', 'chest_pain', 'shortness_of_breath']

# Sample data: each row corresponds to symptoms in order above (1 = symptom present, 0 = absent)
X = np.array([
    [1, 1, 0, 0],  # fever + cough -> flu
    [0, 0, 1, 1],  # chest pain + shortness of breath -> heart disease
    [0, 0, 0, 0],  # no symptoms -> common cold
])

y = ['flu', 'heart_disease', 'common_cold']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and symptom order to disk
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'symptom_order': symptom_order}, f)

print("Model trained and saved successfully.")
