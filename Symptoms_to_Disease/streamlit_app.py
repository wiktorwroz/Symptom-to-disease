# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import joblib
from pathlib import Path

st.set_page_config(page_title="Symptoms → Disease → Doctor", page_icon="🩺")
st.title("🩺 Symptoms → Disease → Doctor")
st.write("Wybierz gotowe zestawy objawow z listy, a aplikacja pokaze sugerowanych lekarzy.")

def clean_symptoms(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def comma_tokenizer(text):
    return [t.strip() for t in str(text).split(",") if t.strip()]

model_path = Path("model.joblib")
vec_path = Path("vectorizer.joblib")
MIN_VEC_SIZE = 1000
MAX_MODEL_SIZE = 500 * 1024 * 1024  # 500 MB

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    mapping_path = Path("disease_to_doctor.joblib")
    d2d = joblib.load(mapping_path) if mapping_path.exists() else {}
    return model, vectorizer, d2d

if model_path.exists() and vec_path.exists() and vec_path.stat().st_size >= MIN_VEC_SIZE:
    if model_path.stat().st_size > MAX_MODEL_SIZE:
        st.warning(
            f"model.joblib jest za duzy ({model_path.stat().st_size / (1024**3):.2f} GB). "
            "Wybierz lzejszy model i zapisz ponownie."
        )
        st.stop()

    with st.spinner("Ladowanie modelu..."):
        model, vectorizer, DISEASE_TO_DOCTOR = load_artifacts()
    mapping_path = Path("disease_to_doctor.joblib")
    def get_doctor(d):
        d = str(d).lower()
        for k, v in DISEASE_TO_DOCTOR.items():
            if k in d: return v
        return "Lekarz rodzinny"

    symptom_groups = {
        "Skora: swedzenie, wysypka, zaczerwienienie, goraczka": "itching, skin rash, redness, fever",
        "Uklad oddechowy: kaszel, goraczka, dusznosc, bol gardla": "cough, fever, breathlessness, sore throat",
        "Zoladek i jelita: nudnosci, wymioty, bol brzucha, biegunka": "nausea, vomiting, abdominal pain, diarrhea",
        "Neurologia: bol glowy, zawroty glowy, nudnosci, wrazliwosc na swiatlo": "headache, dizziness, nausea, sensitivity to light",
        "Laryngologia: bol gardla, katar, kichanie, chrypka": "sore throat, runny nose, sneezing, hoarseness",
        "Kardiologia: bol w klatce piersiowej, dusznosc, zmeczenie, kolatanie serca": "chest pain, breathlessness, fatigue, palpitations",
        "Urologia: bol przy oddawaniu moczu, czestomocz, goraczka, bol podbrzusza": "burning urination, frequent urination, fever, lower abdominal pain",
        "Ortopedia: bol stawow, obrzek, sztywnosc, ograniczenie ruchu": "joint pain, swelling, stiffness, reduced movement",
        "Endokrynologia: wzmozone pragnienie, czeste oddawanie moczu, oslabienie, spadek masy": "excessive thirst, frequent urination, fatigue, weight loss",
        "Okulistyka: bol oka, zaczerwienienie oka, zamglone widzenie, swiatlowstret": "eye pain, eye redness, blurred vision, photophobia",
    }

    selected_groups = st.multiselect(
        "Wybierz objawy (mozna zaznaczyc kilka zestawow):",
        options=list(symptom_groups.keys()),
    )

    if selected_groups:
        selected_symptom_text = ", ".join(symptom_groups[g] for g in selected_groups)
        cleaned = clean_symptoms(selected_symptom_text)
        st.caption(f"Zlaczone objawy: {cleaned}")

        # Podglad sugerowanych lekarzy na bazie kazdego kliknietego zestawu.
        doctor_suggestions = []
        for group in selected_groups:
            group_cleaned = clean_symptoms(symptom_groups[group])
            try:
                group_pred = model.predict(vectorizer.transform([group_cleaned]))
                if len(group_pred) > 0:
                    doctor_suggestions.append(get_doctor(group_pred[0]))
            except Exception:
                continue

        unique_doctors = sorted(set(doctor_suggestions))
        if unique_doctors:
            st.info("Sugerowani lekarze na podstawie zaznaczonych zestawow: " + ", ".join(unique_doctors))

        if st.button("Przewidz glowna chorobe"):
            try:
                X = vectorizer.transform([cleaned])
                pred = model.predict(X)
                if len(pred) == 0:
                    st.warning("Model nie zwrocil predykcji.")
                else:
                    disease = pred[0]
                    doctor = get_doctor(disease)
                    st.success(f"Przewidywana choroba: **{disease}**")
                    st.info(f"Sugerowany lekarz: **{doctor}**")
            except Exception as e:
                st.error(f"Blad predykcji: {e}")
    else:
        st.warning("Wybierz przynajmniej jeden zestaw objawow z listy.")
else:
    if vec_path.exists() and vec_path.stat().st_size < MIN_VEC_SIZE:
        st.warning("vectorizer.joblib uszkodzony lub pusty. Uruchom notebook od sekcji Feature Engineering i komórkę z joblib.dump.")
    else:
        st.warning("Zapisz model i vectorizer (joblib) w tym samym katalogu. W notebooku: Run od Feature Engineering, potem joblib.dump.")
