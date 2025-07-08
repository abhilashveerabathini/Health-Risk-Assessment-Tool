
import joblib
import numpy as np

def adjust_risk_by_age(risk, age):
    if age >= 60:
        return min(risk + 0.1, 1.0)
    elif age <= 25:
        return max(risk - 0.05, 0.0)
    return risk

def get_final_risk_scores(cvd_prob, heart_prob, obesity_prob, age):
    return {
        "CVD": adjust_risk_by_age(cvd_prob, age),
        "Heart": adjust_risk_by_age(heart_prob, age),
        "Obesity": adjust_risk_by_age(obesity_prob, age)
    }

def prioritize_recommendations(risks):
    sorted_risks = sorted(risks.items(), key=lambda x: -x[1])
    recs = []
    for condition, score in sorted_risks:
        if condition == "CVD" and score > 0.6:
            recs.append("⚠️ High CVD risk: Consult cardiologist, reduce sodium, start walking daily.")
        elif condition == "Heart" and score > 0.5:
            recs.append("❤️ Heart disease risk: Manage stress, monitor BP, eat heart-healthy foods.")
        elif condition == "Obesity" and score > 0.5:
            recs.append("⚖️ Obesity risk: Reduce calorie intake, increase activity, avoid sugar.")
    return recs

def test_user_risk(user_input):
    cvd_model = joblib.load("cvd_model.pkl")
    heart_model = joblib.load("heart_model.pkl")
    obesity_model = joblib.load("obesity_model.pkl")
    scaler_cvd = joblib.load("scaler_cvd.pkl")
    scaler_heart = joblib.load("scaler_heart.pkl")
    scaler_obesity = joblib.load("scaler_obesity.pkl")

    cvd_input = [user_input["age"], user_input["gender"], user_input["smoking"],
                 user_input["alcohol"], user_input["activity"], user_input["heredity"]]
    heart_input = cvd_input.copy()
    obesity_input = [user_input["bmi"], user_input["activity"], user_input["heredity"]]

    cvd_prob = cvd_model.predict_proba(scaler_cvd.transform([cvd_input]))[0][1]
    heart_prob = heart_model.predict_proba(scaler_heart.transform([heart_input]))[0][1]
    obesity_prob = obesity_model.predict_proba(scaler_obesity.transform([obesity_input]))[0][1]

    risk_scores = get_final_risk_scores(cvd_prob, heart_prob, obesity_prob, user_input["age"])
    recommendations = prioritize_recommendations(risk_scores)

    print("📊 Adjusted Risk Scores:")
    for k, v in risk_scores.items():
        print(f"{k}: {v:.2f}")

    print("\n📝 Recommendations:")
    if recommendations:
        for r in recommendations:
            print(" -", r)
    else:
        print("✅ All risks are currently low. Maintain healthy lifestyle.")

user_input = {
    "age": 52,
    "gender": 1,
    "smoking": 1,
    "alcohol": 0,
    "activity": 0,
    "bmi": 33.2,
    "heredity": 1
}

test_user_risk(user_input)
