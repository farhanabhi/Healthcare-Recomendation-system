from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import shap

app = Flask(__name__)

# load dataset
df = pd.read_csv("disease_data.csv", encoding='latin1')

# get unique symptoms from dataset
symptoms = df["symptom1"].unique().tolist() + df["symptom2"].unique().tolist() + df["symptom3"].unique().tolist()

# create a dictionary to map symptoms to disease indices
symptom_to_disease = {}
for i, row in df.iterrows():
    for symptom in [row["symptom1"], row["symptom2"], row["symptom3"]]:
        if symptom not in symptom_to_disease:
            symptom_to_disease[symptom] = []
        symptom_to_disease[symptom].append(i)

# create a one-hot encoder
ohe = OneHotEncoder(handle_unknown='ignore')

# fit the one-hot encoder to the symptoms
X = pd.DataFrame({'symptom1': df['symptom1'], 'symptom2': df['symptom2'], 'symptom3': df['symptom3']})
X_encoded = ohe.fit_transform(X)

# create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
y = df["disease"]
rfc.fit(X_encoded, y)

# create a SHAP explainer with feature perturbation
explainer = shap.TreeExplainer(rfc, feature_perturbation='interventional')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptom1 = data["symptom1"]
    symptom2 = data["symptom2"]
    symptom3 = data["symptom3"]

    # encode the symptoms using the one-hot encoder
    X_pred = pd.DataFrame({'symptom1': [symptom1], 'symptom2': [symptom2], 'symptom3': [symptom3]})
    X_pred_encoded = ohe.transform(X_pred).toarray()  # Convert to dense array

    # predict the disease using the random forest classifier
    y_pred = rfc.predict(X_pred_encoded)[0]

    # find the index of the row in df that corresponds to the predicted disease
    idx = df[df['disease'] == y_pred].index[0]

    # get the disease information from the dataset
    disease_info = df.loc[idx, :]  # Select the entire row

    causes = disease_info["causes"]
    remedies = disease_info["remedies"]
    dietary_recommendations = disease_info["dietary_recommendations"]

    # get the SHAP values for the prediction
    shap_values = explainer.shap_values(X_pred_encoded, check_additivity=False)

    # return the prediction result
    return jsonify({
        "disease": y_pred,
        "causes": causes,
        "remedies": remedies,
        "dietary_recommendations": dietary_recommendations,
        "shap_values": shap_values.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)