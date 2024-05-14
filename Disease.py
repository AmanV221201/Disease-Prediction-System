import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Define symptoms and diseases
l1 = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
        'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
        'Migraine','Cervical spondylosis',
        'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
        'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
        'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
        'Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
        'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
        'Impetigo']

# Read and process training data
df = pd.read_csv("Training.csv")
df['prognosis'] = df['prognosis'].str.strip()  # Remove any leading or trailing whitespace
df.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
X = df[l1]
y = df['prognosis'].astype(int)

# Read and process testing data
tr = pd.read_csv("Testing.csv")
tr['prognosis'] = tr['prognosis'].str.strip()  # Remove any leading or trailing whitespace
tr.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
X_test = tr[l1]
y_test = tr['prognosis'].astype(int)

# Initialize and train the model
gnb = MultinomialNB()
gnb.fit(X, y)

# Streamlit app
st.title("Disease Prediction from Symptoms")

st.write("Enter the symptoms to predict the disease:")

symptom1 = st.selectbox("Symptom 1", ["None"] + l1)
symptom2 = st.selectbox("Symptom 2", ["None"] + l1)
symptom3 = st.selectbox("Symptom 3", ["None"] + l1)
symptom4 = st.selectbox("Symptom 4", ["None"] + l1)
symptom5 = st.selectbox("Symptom 5", ["None"] + l1)

if st.button("Predict"):
    if symptom1 == "None" and symptom2 == "None" and symptom3 == "None" and symptom4 == "None" and symptom5 == "None":
        st.warning("Please enter at least one symptom")
    else:
        psymptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
        l2 = [1 if symptom in psymptoms else 0 for symptom in l1]
        inputtest = np.array(l2).reshape(1, -1)
        predict = gnb.predict(inputtest)
        predicted_disease = disease[predict[0]]
        st.write(f"The predicted disease is: {predicted_disease}")

# Display model accuracy
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")
