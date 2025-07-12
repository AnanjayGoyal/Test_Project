import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- 1. Load the dataset ---
# Assuming 'Disease-Symptom Dataset.csv' is in the same directory as this script.
# You can download it from Kaggle: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
try:
    df = pd.read_csv('Disease-Symptom Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Disease-Symptom Dataset.csv' not found.")
    print("Please download the dataset from https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset")
    print("and place it in the same directory as this Python script.")
    exit()

# --- 2. Preprocessing ---
# Based on the previous error output, the disease column is named 'diseases'.
# Check if the 'diseases' column exists. If not, print available columns for debugging.
if 'diseases' not in df.columns:
    print("\nError: The 'diseases' column was not found in the dataset.")
    print("Available columns are:", df.columns.tolist())
    print("Please ensure your dataset has a column named 'diseases' or update the code to use the correct column name.")
    exit()

# --- Filter out rare diseases (New Addition for efficiency) ---
# This helps reduce the number of unique classes, which speeds up training
# and improves model performance for the remaining, more common diseases.
min_disease_occurrences = 50 # Increased threshold to further reduce unique classes.
                               # Diseases appearing fewer times than this will be removed.

original_num_diseases = df['diseases'].nunique()
disease_counts = df['diseases'].value_counts()
diseases_to_keep = disease_counts[disease_counts >= min_disease_occurrences].index
df = df[df['diseases'].isin(diseases_to_keep)].reset_index(drop=True)

new_num_diseases = df['diseases'].nunique()
print(f"Filtered dataset: Reduced from {original_num_diseases} unique diseases to {new_num_diseases} unique diseases.")
print(f"Remaining samples after filtering: {len(df)} rows.")


# The 'diseases' column is the target, and the rest are symptoms.
X = df.drop('diseases', axis=1)
y = df['diseases']

# Get all available symptoms from the dataset
all_symptoms = X.columns.tolist()

# Define a mapping for common symptom aliases to their standardized names in the dataset
# This helps in recognizing variations of symptoms entered by the user.
symptom_alias_map = {
    "body pain": "muscle_pain",
    "vomit": "vomiting",
    "stomach ache": "stomach_pain",
    "sore eyes": "eye_redness",
    "runny nose": "coryza",
    "stuffy nose": "nasal_congestion",
    "joint ache": "joint_pain",
    "chills and fever": "chills", # Assuming fever is often entered separately
    "coughing": "cough",
    "headache": "headache",
    "fever": "fever",
    "fatigue": "fatigue",
    "nausea": "nausea",
    "diarrhea": "diarrhea",
    "skin rash": "skin_rash",
    "sore throat": "sore_throat",
    "chest pain": "chest_pain",
    "dizziness": "dizziness",
    "insomnia": "insomnia",
    "weight loss": "weight_loss",
    "weight gain": "weight_gain",
    "shortness of breath": "shortness_of_breath",
    "sweating": "sweating",
    "yellowish skin": "jaundice",
    "dark urine": "dark_urine",
    "loss of appetite": "loss_of_appetite",
    "back pain": "back_pain",
    "neck pain": "neck_pain",
    "constipation": "constipation",
    "abdominal pain": "abdominal_pain",
    "muscle weakness": "muscle_weakness",
    "irritability": "irritability",
    "depression": "depression",
    "anxiety": "anxiety_and_nervousness",
    "chills": "chills"
}


# Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 3. Model Training ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Starting model training... This might take a while for large datasets.")
# Initialize and train the SGDClassifier
# SGDClassifier is highly efficient for large datasets as it processes data incrementally.
# 'loss='log_loss'' makes it equivalent to Logistic Regression.
# 'max_iter' controls the number of passes over the training data.
model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, n_jobs=-1, tol=1e-3)
model.fit(X_train, y_train)

print(f"Model trained with accuracy on test set: {model.score(X_test, y_test):.2f}")

# --- 4. Prediction Function ---
def predict_disease(symptoms_input, all_symptoms, model, label_encoder, symptom_alias_map):
    """
    Predicts a disease based on a list of symptoms.

    Args:
        symptoms_input (list): A list of symptoms provided by the user (strings).
        all_symptoms (list): A list of all possible symptoms known by the model (from the dataset).
        model (sklearn.linear_model.SGDClassifier): The trained machine learning model.
        label_encoder (sklearn.preprocessing.LabelEncoder): The encoder used for disease labels.
        symptom_alias_map (dict): A dictionary mapping common symptom aliases to standardized names.

    Returns:
        str: The predicted disease name, or an error message if no symptoms are recognized,
             or a message indicating the disease was filtered out.
    """
    # Create a zero-filled array for the input symptoms, matching the model's feature shape
    input_data = pd.DataFrame(0, index=[0], columns=all_symptoms)
    recognized_symptoms = []

    # Map user input symptoms to the model's symptom vector
    for symptom in symptoms_input:
        symptom_cleaned = symptom.strip().lower() # Clean user input

        # Check for direct match or alias match
        mapped_symptom = symptom_alias_map.get(symptom_cleaned, symptom_cleaned.replace(" ", "_"))

        if mapped_symptom in all_symptoms:
            input_data[mapped_symptom] = 1
            recognized_symptoms.append(symptom.strip())
        else:
            print(f"Warning: Symptom '{symptom.strip()}' not recognized and will be ignored.")

    if not recognized_symptoms:
        return "No recognized symptoms provided. Please try again with valid symptoms."
    else:
        print(f"Symptoms used for prediction: {', '.join(recognized_symptoms)}")

    # Predict the disease
    prediction_encoded = model.predict(input_data)
    predicted_disease = label_encoder.inverse_transform(prediction_encoded)

    # Check if the predicted disease was one of the filtered-out diseases
    # This check is more complex as the model only knows about the non-filtered diseases.
    # We can only predict from the labels the model was trained on.
    return predicted_disease[0]

# --- 5. User Interaction ---
if __name__ == "__main__":
    print("\n--- Disease Prediction Program ---")
    print("Enter your symptoms separated by commas (e.g., fever, cough, headache).")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter your symptoms: ").strip()
        if user_input.lower() == 'exit':
            break

        if not user_input:
            print("Please enter some symptoms.")
            continue

        symptoms_list = [s.strip() for s in user_input.split(',')]

        # Predict and display
        predicted = predict_disease(symptoms_list, all_symptoms, model, label_encoder, symptom_alias_map)
        print(f"\nBased on your symptoms, the predicted disease is: {predicted}")

    print("\nThank you for using the Disease Prediction Program!")
