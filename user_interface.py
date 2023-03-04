import numpy as np
import pickle

# Load the trained model from a file
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Ask for patient data
pregnancies = int(input('Number of pregnancies: '))
glucose = int(input('Glucose level: '))
blood_pressure = int(input('Blood pressure: '))
skin_thickness = int(input('Skin thickness: '))
insulin = int(input('Insulin level: '))
bmi = float(input('BMI: '))
age = int(input('Age: '))

# Create a numpy array with the patient's data
patient_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age]])

# Make a prediction for the patient
prediction = model.predict(patient_data)

# Print the prediction
if prediction == 0:
    print('The patient does not have diabetes.')
else:
    print('The patient has diabetes.')
