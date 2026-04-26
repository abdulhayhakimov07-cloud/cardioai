from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
import json
from datetime import datetime
import urllib.request

app = Flask(__name__)

# Dataset
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores[name] = round(accuracy_score(y_test, y_pred) * 100, 2)

best_name = max(scores, key=scores.get)
best_model = models[best_name]

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Grafiklar
def create_charts():
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ee5a24', '#f9ca24', '#6ab04c']
    bars = ax.bar(scores.keys(), scores.values(), color=colors, width=0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Aniqlik (%)')
    ax.set_title('Modellar Taqqoslash')
    for bar, score in zip(bars, scores.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/model_chart.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df['target'].value_counts()
    ax.pie(counts, labels=['Soglom', 'Xasta'],
           colors=['#6ab04c', '#ee5a24'],
           autopct='%1.1f%%', startangle=90)
    ax.set_title("Bemorlar Taqsimoti")
    plt.tight_layout()
    plt.savefig('static/disease_chart.png')
    plt.close()

create_charts()

# Tavsiyalar bazasi
def get_advice(risk):
    if risk >= 80:
        return {
            'daraja': 'JUDA YUQORI XAVF',
            'rang': 'danger',
            'tavsiya': [
                '🚨 Zudlik bilan kardiolog shifokorga murojaat qiling!',
                '🏥 Shoshilinch tarzda kasalxonaga boring',
                '💊 Shifokor ko\'rsatmasiz dori ichmang',
                '🛌 To\'liq dam oling, jismoniy faollikni to\'xtating',
                '📞 Yaqinlaringizni xabardor qiling',
            ],
            'operatsiya': 'Stent o\'rnatish yoki shunting operatsiyasi talab etilishi mumkin',
            'dorilar': 'Aspirin, Beta-blokatorlar, Nitratlar (faqat shifokor ko\'rsatmasi bilan)'
        }
    elif risk >= 60:
        return {
            'daraja': 'YUQORI XAVF',
            'rang': 'warning',
            'tavsiya': [
                '⚠️ 24 soat ichida shifokorga boring',
                '📋 EKG va qon tahlili topshiring',
                '🚭 Chekishni darhol tashlang',
                '🥗 Yog\'li ovqatlardan saqlaning',
                '🧘 Stress va zo\'riqishdan uzoq yuring',
            ],
            'operatsiya': 'Angiografiya tekshiruvi tavsiya etiladi',
            'dorilar': 'Statinlar, ACE inhibitorlar (shifokor ko\'rsatmasi bilan)'
        }
    elif risk >= 40:
        return {
            'daraja': 'O\'RTA XAVF',
            'rang': 'medium',
            'tavsiya': [
                '📅 Hafta ichida shifokor ko\'rigidan o\'ting',
                '🏃 Kunlik 30 daqiqa yurish',
                '🥦 Ko\'proq sabzavot va meva iste\'mol qiling',
                '💧 Kuniga 2 litr suv iching',
                '⚖️ Vazningizni nazorat qiling',
            ],
            'operatsiya': 'Hozircha operatsiya kerak emas, kuzatuv tavsiya etiladi',
            'dorilar': 'Omega-3, Vitamin D (shifokor maslahatidan so\'ng)'
        }
    else:
        return {
            'daraja': 'PAST XAVF',
            'rang': 'safe',
            'tavsiya': [
                '✅ Sog\'lom holatdasiz, shunday davom eting!',
                '🏋️ Muntazam jismoniy mashqlar qiling',
                '🥗 Muvozanatli ovqatlaning',
                '😴 Kuniga 7-8 soat uxlang',
                '🧪 Yiliga 1 marta profilaktik tekshiruv o\'ting',
            ],
            'operatsiya': 'Operatsiya talab etilmaydi',
            'dorilar': 'Dori kerak emas, sog\'lom turmush tarzi yetarli'
        }

HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:50]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)

@app.route('/')
def home():
    history = load_history()
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Noma\'lum')
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]

    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal']
    df_input = pd.DataFrame([features], columns=columns)

    prediction = best_model.predict(df_input)[0]
    proba = best_model.predict_proba(df_input)[0]
    risk = round(proba[1] * 100, 1)

    result = 'XAVF BOR' if prediction == 1 else 'SOGLOM'
    advice = get_advice(risk)

    record = {
        'ism': name,
        'sana': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'yosh': int(features[0]),
        'jins': 'Erkak' if features[1] == 1 else 'Ayol',
        'natija': result,
        'xavf': risk,
        'aniqlik': scores[best_name]
    }
    save_history(record)

    return render_template('result.html',
        name=name,
        result=result,
        risk=risk,
        accuracy=scores[best_name],
        advice=advice,
        features=dict(zip(columns, features))
    )

@app.route('/coach', methods=['POST'])
def coach():
    data = request.get_json()
    user_msg = data.get('message', '')

    try:
        import urllib.request
        import json as jsonlib

        API_KEY = "AIzaSyB7n_Qq6ZSxV6KJwGEUQAplVzdq095p_8c"  # <- shu yerga o'zingizning keyni qo'ying

        payload = jsonlib.dumps({
            "contents": [{
                "parts": [{
                    "text": f"Siz yurak kasalliklari bo'yicha mutaxassis shifokor-maslahatchi CardioAI siz. Faqat sog'liq, yurak kasalliklari, dorilar, turmush tarzi haqida maslahat bering. O'zbek tilida qisqa, aniq va tushunarli javob bering. Har doim oxirida 'Batafsil uchun real shifokorga murojaat qiling' deb qo'shing.\n\nSavol: {user_msg}"
                }]
            }]
        }).encode('utf-8')

        req = urllib.request.Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}',
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            result = jsonlib.loads(response.read())
            reply = result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        reply = "Kechirasiz, hozir javob bera olmayapman. Iltimos qayta urinib ko'ring yoki real shifokorga murojaat qiling."

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)