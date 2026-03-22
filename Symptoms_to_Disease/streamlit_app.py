# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import joblib
from pathlib import Path

st.set_page_config(page_title="Symptoms → Disease → Doctor", page_icon="🩺")
st.title("🩺 Symptoms → Disease → Doctor")
st.write("Wybierz pojedyncze objawy z pelnej listy, a aplikacja zasugeruje chorobe i lekarza.")

def clean_symptoms(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def comma_tokenizer(text):
    return [t.strip() for t in str(text).split(",") if t.strip()]

Disease_TRANSLATIONS_PL = {
    "fungal infection": "infekcja grzybicza",
    "allergy": "alergia",
    "gerd": "choroba refluksowa przelyku (GERD)",
    "chronic cholestasis": "przewlekla cholestaza",
    "drug reaction": "reakcja polekowa",
    "peptic ulcer diseae": "choroba wrzodowa zoladka/dwunastnicy",
    "aids": "AIDS",
    "diabetes": "cukrzyca",
    "gastroenteritis": "zapalenie zoladka i jelit",
    "bronchial asthma": "astma oskrzelowa",
    "hypertension": "nadcisnienie tetnicze",
    "migraine": "migrena",
    "cervical spondylosis": "spondyloza szyjna",
    "paralysis brain hemorrhage": "porazenie po krwotoku mozgowym",
    "jaundice": "zoltaczka",
    "malaria": "malaria",
    "chicken pox": "ospa wietrzna",
    "dengue": "denga",
    "typhoid": "dur brzuszny",
    "hepatitis a": "wirusowe zapalenie watroby typu A",
    "hepatitis b": "wirusowe zapalenie watroby typu B",
    "hepatitis c": "wirusowe zapalenie watroby typu C",
    "hepatitis d": "wirusowe zapalenie watroby typu D",
    "hepatitis e": "wirusowe zapalenie watroby typu E",
    "alcoholic hepatitis": "alkoholowe zapalenie watroby",
    "tuberculosis": "gruzlica",
    "common cold": "przeziebienie",
    "pneumonia": "zapalenie pluc",
    "dimorphic hemmorhoids piles": "hemoroidy",
    "heart attack": "zawal serca",
    "varicose veins": "zylaki",
    "hypothyroidism": "niedoczynnosc tarczycy",
    "hyperthyroidism": "nadczynnosc tarczycy",
    "hypoglycemia": "hipoglikemia",
    "osteoarthristis": "choroba zwyrodnieniowa stawow",
    "arthritis": "zapalenie stawow",
    "acne": "tradzik",
    "urinary tract infection": "zakazenie drog moczowych",
    "psoriasis": "luszczyca",
    "impetigo": "liszajec zakazny",
}

def translate_disease_to_polish(disease_name):
    key = str(disease_name).lower().strip()
    key = re.sub(r"[_-]+", " ", key)
    key = re.sub(r"\s+", " ", key)
    return Disease_TRANSLATIONS_PL.get(key, None)

DISEASE_HELPER_SYMPTOMS = {
    "migraine": ["nausea", "vomiting", "sensitivity to light"],
    "common cold": ["runny nose", "sneezing", "throat irritation"],
    "pneumonia": ["chest pain", "breathlessness", "high fever"],
    "bronchial asthma": ["breathlessness", "cough", "chest tightness"],
    "urinary tract infection": ["burning urination", "frequent urination", "foul smell of urine"],
    "dengue": ["high fever", "joint pain", "skin rash"],
    "chicken pox": ["skin rash", "itching", "high fever"],
    "gastroenteritis": ["diarrhea", "vomiting", "abdominal pain"],
}

def normalize_text(value):
    value = str(value).lower().strip()
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value

SYMPTOM_TRANSLATIONS_PL = {
    "itching": "swedzenie",
    "skin rash": "wysypka skorna",
    "redness": "zaczerwienienie",
    "fever": "goraczka",
    "high fever": "wysoka goraczka",
    "cough": "kaszel",
    "sore throat": "bol gardla",
    "runny nose": "katar",
    "sneezing": "kichanie",
    "hoarseness": "chrypka",
    "breathlessness": "dusznosc",
    "chest pain": "bol w klatce piersiowej",
    "chest tightness": "ucisk w klatce piersiowej",
    "fatigue": "zmeczenie",
    "palpitations": "kolatanie serca",
    "nausea": "nudnosci",
    "vomiting": "wymioty",
    "abdominal pain": "bol brzucha",
    "diarrhea": "biegunka",
    "headache": "bol glowy",
    "dizziness": "zawroty glowy",
    "sensitivity to light": "nadwrazliwosc na swiatlo",
    "joint pain": "bol stawow",
    "swelling": "obrzek",
    "stiffness": "sztywnosc",
    "reduced movement": "ograniczenie ruchu",
    "burning urination": "pieczenie przy oddawaniu moczu",
    "frequent urination": "czestomocz",
    "lower abdominal pain": "bol podbrzusza",
    "foul smell of urine": "nieprzyjemny zapach moczu",
    "eye pain": "bol oka",
    "eye redness": "zaczerwienienie oka",
    "blurred vision": "zamglone widzenie",
    "photophobia": "swiatlowstret",
    "weight loss": "spadek masy ciala",
    "excessive thirst": "wzmozone pragnienie",
    "throat irritation": "podraznienie gardla",
}

SYMPTOM_WORD_TRANSLATIONS_PL = {
    "pain": "bol",
    "fever": "goraczka",
    "high": "wysoka",
    "abdominal": "brzucha",
    "chest": "klatki",
    "joint": "stawow",
    "eye": "oka",
    "redness": "zaczerwienienie",
    "burning": "pieczenie",
    "frequent": "czeste",
    "urination": "oddawanie moczu",
    "urine": "mocz",
    "foul": "nieprzyjemny",
    "smell": "zapach",
    "runny": "wodnisty",
    "nose": "nos",
    "sore": "bolacy",
    "throat": "gardlo",
    "skin": "skora",
    "rash": "wysypka",
    "blurred": "zamglone",
    "vision": "widzenie",
    "sensitivity": "nadwrazliwosc",
    "light": "swiatlo",
    "weight": "masa",
    "loss": "spadek",
}

def translate_symptom_to_polish(symptom_name):
    symptom_en = normalize_text(symptom_name)
    if symptom_en in SYMPTOM_TRANSLATIONS_PL:
        return SYMPTOM_TRANSLATIONS_PL[symptom_en]

    translated_tokens = []
    translated_any = False
    for token in symptom_en.split(" "):
        translated = SYMPTOM_WORD_TRANSLATIONS_PL.get(token, token)
        if translated != token:
            translated_any = True
        translated_tokens.append(translated)

    if translated_any:
        return " ".join(translated_tokens)
    return None

def display_symptom(symptom_name):
    return normalize_text(symptom_name)

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

    all_symptoms = sorted([str(s) for s in vectorizer.get_feature_names_out()])
    all_symptoms_set = set(all_symptoms)

    st.subheader("Krok 1: Dodawanie objawow pojedynczo")
    st.caption("Wybieraj objawy jeden po drugim. Mozesz je potem usunac.")

    if "selected_symptoms" not in st.session_state:
        st.session_state.selected_symptoms = []

    symptom_to_add = st.selectbox(
        "Wybierz objaw do dodania:",
        options=all_symptoms,
        format_func=display_symptom,
    )
    if st.button("Dodaj objaw"):
        if symptom_to_add not in st.session_state.selected_symptoms:
            st.session_state.selected_symptoms.append(symptom_to_add)

    st.session_state.selected_symptoms = st.multiselect(
        "Wybrane objawy pacjenta:",
        options=all_symptoms,
        default=st.session_state.selected_symptoms,
        format_func=display_symptom,
    )

    if st.session_state.selected_symptoms:
        cleaned = clean_symptoms(", ".join(st.session_state.selected_symptoms))
        st.caption("Aktualny zestaw objawow: " + ", ".join(display_symptom(s) for s in st.session_state.selected_symptoms))

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

                    # Pytanie pomocnicze: dopytaj o dodatkowy typowy objaw.
                    disease_key = normalize_text(disease)
                    helper_candidates = DISEASE_HELPER_SYMPTOMS.get(disease_key, [])
                    missing_candidate = None
                    selected_norm = {normalize_text(s) for s in st.session_state.selected_symptoms}
                    for candidate in helper_candidates:
                        c_norm = normalize_text(candidate)
                        if c_norm not in selected_norm:
                            if candidate in all_symptoms_set:
                                missing_candidate = candidate
                                break
                            candidate_alt = candidate.replace(" ", "_")
                            if candidate_alt in all_symptoms_set:
                                missing_candidate = candidate_alt
                                break

                    if missing_candidate:
                        st.markdown("---")
                        st.subheader("Pytanie pomocnicze")
                        st.write(f"Czy wystepuje tez objaw: **{display_symptom(missing_candidate)}** ?")
                        helper_answer = st.radio(
                            "Odpowiedz:",
                            options=["Nie", "Tak"],
                            horizontal=True,
                            key=f"helper_{normalize_text(missing_candidate)}",
                        )
                        if helper_answer == "Tak":
                            if missing_candidate not in st.session_state.selected_symptoms:
                                st.session_state.selected_symptoms.append(missing_candidate)
                                st.success("Dodano objaw. Kliknij ponownie 'Przewidz glowna chorobe', aby odswiezyc wynik.")
            except Exception as e:
                st.error(f"Blad predykcji: {e}")
    else:
        st.warning("Dodaj przynajmniej jeden objaw z listy.")
else:
    if vec_path.exists() and vec_path.stat().st_size < MIN_VEC_SIZE:
        st.warning("vectorizer.joblib uszkodzony lub pusty. Uruchom notebook od sekcji Feature Engineering i komórkę z joblib.dump.")
    else:
        st.warning("Zapisz model i vectorizer (joblib) w tym samym katalogu. W notebooku: Run od Feature Engineering, potem joblib.dump.")
