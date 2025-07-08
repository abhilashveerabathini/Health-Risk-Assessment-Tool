# Health-Risk-Assessment-Tool
# Personalized Health Risk Assessment Tool

This project predicts individual health risks using machine learning models trained on clinical and lifestyle data. It evaluates three key risk areas:

- **Cardiovascular Disease (CVD) Risk**
- **Heart Disease Risk**
- **Obesity Risk**

## 🔍 How It Works

1. Takes user inputs including:
   - Age, Gender
   - Lifestyle factors (smoking, alcohol, activity)
   - BMI and family medical history

2. Uses pre-trained ML models to predict the probability of each risk:
   - CVD: Multi-class classification
   - Heart Disease: Binary classification
   - Obesity: Binary classification

3. Adjusts risk scores based on age
4. Prioritizes personalized health recommendations

## 🏗️ Project Structure

- `main.py`: Main script for running the assessment
- `*.pkl`: Pretrained models and scalers
- `requirements.txt`: Python dependencies

## 🧪 Sample Input

```python
{
  "age": 52,
  "gender": 1,
  "smoking": 1,
  "alcohol": 0,
  "activity": 0,
  "bmi": 33.2,
  "heredity": 1
}
