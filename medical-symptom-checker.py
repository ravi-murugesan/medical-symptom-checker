import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def main():
    st.write('''
    # Medical Symptom checker app
    ''')

    symptoms = []
    
    lst1 = ['abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads', 'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feets', 'coma', 'congestion', 'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea', 'dischromic _patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips', 'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 'fluid_overload.1', 'foul_smell_of urine', 'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'malaise', 
        'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose', 'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 'spinning_movements', 'spotting_ urination', 'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties', 'swollen_legs', 'throat_irritation', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin']
    
    disease_di = {'(vertigo) Paroymsal  Positional Vertigo': 0, 'AIDS': 1, 'Acne': 2, 'Alcoholic hepatitis': 3, 'Allergy': 4, 'Arthritis': 5, 'Bronchial Asthma': 6, 'Cervical spondylosis': 7, 'Chicken pox': 8, 'Chronic cholestasis': 9, 'Common Cold': 10, 'Dengue': 11, 'Diabetes ': 12, 'Dimorphic hemmorhoids(piles)': 13, 'Drug Reaction': 14, 'Fungal infection': 15, 'GERD': 16, 'Gastroenteritis': 17, 'Heart attack': 18, 'Hepatitis B': 19, 'Hepatitis C': 20, 'Hepatitis D': 21, 'Hepatitis E': 22, 'Hypertension ': 23, 'Hyperthyroidism': 24, 'Hypoglycemia': 25, 'Hypothyroidism': 26, 'Impetigo': 27, 'Jaundice': 28, 'Malaria': 29, 'Migraine': 30, 'Osteoarthristis': 31, 'Paralysis (brain hemorrhage)': 32, 'Peptic ulcer diseae': 33, 'Pneumonia': 34, 'Psoriasis': 35, 'Tuberculosis': 36, 'Typhoid': 37, 'Urinary tract infection': 38, 'Varicose veins': 39, 'hepatitis A': 40}


    st.subheader('Disclaimer')
    st.write('This web app is intended for educational purposes only. It is not intended for use in diagnosis/detection of diseases or other medical conditions.')

    st.header('Please select your symptoms :')
    symptoms = st.multiselect('(max-20)',[i for i in lst1 if i not in symptoms])
    st.write('   ')
    st.header('Choose the number of top predictions to list :')
    k = st.slider('(default-3)', 1,5,3)

    if st.button('CHECK'):
        if len(symptoms) > 3:
            df = pd.read_csv('pred.csv',index_col=0)
            value1 = [1 if i in symptoms else 0 for i in lst1]
            row1 = pd.Series(value1,lst1)
            df = df.append([row1])

            new_model = keras.models.load_model('ANN_pred.h5')
            pred = new_model.predict(df)
            pred1 = (np.around(pred,3))*100
            a = pred1.tolist()[0]

            group_dict = np.load('dict_group.npy',allow_pickle='TRUE').item()

            df = pd.DataFrame()
            pred = a.copy()
            df['disease'] = [k for k,v in disease_di.items()]
            df['pred'] = pred
            df['pred'] = df['pred'].apply(lambda x: round(x, 2))
            df = df.sort_values(by=['pred'],ascending=False)
            df = df[:k]
            df = df.sort_values(by=['pred'],ascending=True)
            df = df.drop(df[df['pred']== 0].index)

            df.plot(kind="barh", legend=False, width=0.6,color = 'skyblue')
            for i, (p, pr) in enumerate(zip(df['disease'], df["pred"])):
                plt.text(s=p, x=15, y=i, color="black", verticalalignment="center", size=16)
                plt.text(s=str(pr)+"%", x=5, y=i, color="maroon",
                         verticalalignment="center", horizontalalignment="center", size=10)
            plt.axis("off")
            st.pyplot()

            st.header('Other symptoms associated with predicted diseases are : ')
            lst1 = df['disease'].tolist()
            for i in range(len(lst1)):
                for k,v in group_dict.items():
                    if k == lst1[i]:
                        lst2 = v
                other_sym = [x for x in lst2 if x not in symptoms]
                st.write('      ')
                st.header(lst1[i] + ' : ')
                st.write(',    '.join(other_sym))
                st.write('      ')

        else:
            st.write('Please rerun the app and select a minimum of 5 symptoms')


if __name__ == "__main__":
    main()

