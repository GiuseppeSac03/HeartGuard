# app.py

import streamlit as st
import pandas as pd
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Configurazione della pagina
st.set_page_config(
    page_title="Predictor di Problemi Cardiaci",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Titolo e descrizione
st.title("❤️ Predictor di Problemi Cardiaci")
st.markdown("""
Questa applicazione prevede la presenza di problemi al cuore basandosi sui dati inseriti.
Inserisci i tuoi dati personali nella barra laterale per ottenere una previsione.
""")

# Aggiungi un'immagine relativa alla salute del cuore
st.image(
    "https://images.unsplash.com/photo-1580281657521-0d9eab53e55a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=60",
    caption="Prenditi cura del tuo cuore!", 
    use_container_width=True
)

# Funzioni di calcolo
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

def calculate_bp_ratio(ap_hi, ap_lo):
    if ap_lo == 0:
        return 0
    return round(ap_hi / ap_lo, 2)

# Barra laterale per input
st.sidebar.header("Inserisci i tuoi dati")

with st.sidebar:
    # Età
    age_years = st.number_input("🧓 Età (in anni)", min_value=0, max_value=120, value=40, step=1)
    
    # Genere
    gender = st.selectbox("⚧ Genere", options=["Donna (1)", "Uomo (2)"])
    gender = 1 if gender.startswith("Donna") else 2
    
    # Altezza e Peso con un layout a colonne
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("📏 Altezza (in cm)", min_value=50, max_value=250, value=170, step=1)
    with col2:
        weight = st.number_input("⚖️ Peso (in kg)", min_value=10, max_value=300, value=70, step=1)
    
    # Calcolo del BMI
    BMI = calculate_bmi(weight, height)
    
    # Pressione Arteriosa
    st.subheader("🩺 Pressione Arteriosa")
    ap_hi = st.number_input("Sistolica (ap_hi)", min_value=50, max_value=250, value=120, step=1)
    ap_lo = st.number_input("Diastolica (ap_lo)", min_value=30, max_value=150, value=80, step=1)
    
    # Calcolo del rapporto pressione sanguigna
    bp_ratio = calculate_bp_ratio(ap_hi, ap_lo)
    
    # Validazione pressione arteriosa
    if ap_hi <= ap_lo:
        st.error("La pressione sistolica deve essere maggiore della diastolica.")
    
    # Livelli di Colesterolo e Glucosio
    cholesterol = st.selectbox("🔬 Livello di Colesterolo",
                               options=["Normale (1)", "Sopra il normale (2)", "Molto sopra il normale (3)"])
    cholesterol = int(cholesterol.split(" ")[-1].strip("()"))
    
    gluc = st.selectbox("🍭 Livello di Glucosio",
                        options=["Normale (1)", "Sopra il normale (2)", "Molto sopra il normale (3)"])
    gluc = int(gluc.split(" ")[-1].strip("()"))
    
    # Abitudini di Vita
    st.subheader("🚭 Abitudini di Vita")
    smoke = st.selectbox("Sei un fumatore?", options=["No (0)", "Sì (1)"])
    smoke = int(smoke.split(" ")[-1].strip("()"))
    
    alco = st.selectbox("Consumi alcol?", options=["No (0)", "Sì (1)"])
    alco = int(alco.split(" ")[-1].strip("()"))
    
    active = st.selectbox("Sei attivo fisicamente?", options=["No (0)", "Sì (1)"])
    active = int(active.split(" ")[-1].strip("()"))
    
    # Bottone per la previsione
    predict_button = st.button("🔮 Calcola Previsione")

# Mostra riepilogo dei dati inseriti nella main area
st.subheader("📊 Riepilogo dei Dati Inseriti")
data = {
    "Genere": ["Donna" if gender == 1 else "Uomo"],
    "Età (anni)": [age_years],
    "Altezza (cm)": [height],
    "Peso (kg)": [weight],
    "BMI": [BMI],
    "Pressione Sistolica (ap_hi)": [ap_hi],
    "Pressione Diastolica (ap_lo)": [ap_lo],
    "Rapporto BP": [bp_ratio],
    "Colesterolo": [cholesterol],
    "Glucosio": [gluc],
    "Fumatore": ["Sì" if smoke == 1 else "No"],
    "Consumo Alcol": ["Sì" if alco == 1 else "No"],
    "Attività Fisica": ["Sì" if active == 1 else "No"],
}
df = pd.DataFrame(data)
st.table(df)

# Caricamento del modello salvato
@st.cache_resource
def load_model():
    try:
        model = joblib.load('svm_cardio_model.pkl')
        return model
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}")
        return None

# Caricamento dello scaler salvato
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except Exception as e:
        st.error(f"Errore nel caricamento dello scaler: {e}")
        return None

model = load_model()
scaler = load_scaler()

# Definizione delle feature numeriche
numerical_features = ['age_years', 'BMI', 'ap_hi', 'ap_lo', 'bp_ratio']

# Caricamento dei dati di addestramento scalati
@st.cache_resource
def load_training_data_scaled():
    try:
        training_data_scaled = pd.read_csv('X_train.csv')
        return training_data_scaled
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati di addestramento scalati: {e}")
        return None

training_data_scaled = load_training_data_scaled()

# Definizione delle feature utilizzate nel modello
model_features = ['gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
                 'smoke', 'alco', 'active', 'BMI', 'bp_ratio', 'age_years']

# Prepara l'explainer di LIME
if training_data_scaled is not None and model is not None:
    # Assicurati che le feature corrispondano
    training_data_scaled = training_data_scaled[model_features]
    
    # Creazione dell'explainer di LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data_scaled.values,
        feature_names=model_features,
        class_names=['No Problemi Cardiaci', 'Problemi Cardiaci'],
        mode='classification'
    )
else:
    explainer = None  # Non sarà possibile generare spiegazioni senza dati di addestramento

# Funzione per generare suggerimenti basati sulle feature
def generate_suggestions(top_features):
    for feature in top_features:
        if feature.startswith("BMI"):
            st.write("- **BMI:** Considera di mantenere o raggiungere un BMI nella fascia normale (18.5 - 24.9) attraverso una dieta equilibrata e attività fisica regolare.")
        elif feature.startswith("bp_ratio"):
            st.write("- **Rapporto BP:** Mantieni la pressione arteriosa sotto controllo seguendo le indicazioni mediche, riducendo l'assunzione di sale e gestendo lo stress.")
        elif feature.startswith("ap_hi"):
            st.write("- **Pressione Arteriosa Sistolica:** Mantieni la pressione sistolica sotto controllo seguendo le indicazioni mediche, riducendo l'assunzione di sale e gestendo lo stress.")
        elif feature.startswith("ap_lo"):
            st.write("- **Pressione Arteriosa Diastolica:** Mantieni la pressione diastolica sotto controllo seguendo le indicazioni mediche, riducendo l'assunzione di sale e gestendo lo stress.")
        elif feature.startswith("age_years"):
            st.write("- **Età:** L'età è un fattore non modificabile, ma puoi concentrarti sugli altri aspetti per ridurre il rischio.")
        elif feature.startswith("gender"):
            st.write("- **Genere:** Alcuni fattori di rischio possono variare in base al genere. Consulta un medico per consigli personalizzati.")
        elif feature.startswith("cholesterol"):
            st.write("- **Colesterolo:** Monitora e gestisci i livelli di colesterolo attraverso una dieta sana e, se necessario, farmaci prescritti dal medico.")
        elif feature.startswith("gluc"):
            st.write("- **Glucosio:** Mantieni i livelli di glucosio nel sangue controllati con una dieta equilibrata e attività fisica.")
        elif feature.startswith("smoke"):
            st.write("- **Fumo:** Se sei un fumatore, considera di smettere di fumare per ridurre significativamente il rischio di problemi cardiaci.")
        elif feature.startswith("alco"):
            st.write("- **Consumo di Alcol:** Limita il consumo di alcol per migliorare la salute del cuore.")
        elif feature.startswith("active"):
            st.write("- **Attività Fisica:** Aumenta il livello di attività fisica per migliorare la salute cardiovascolare.")
        # Aggiungi ulteriori condizioni per altre feature se necessario

# Gestione della previsione
if predict_button:
    if model and scaler:
        if ap_hi > ap_lo:
            # Preparazione dei dati per la previsione
            input_data = pd.DataFrame({
                "gender": [gender],
                "ap_hi": [ap_hi],
                "ap_lo": [ap_lo],
                "cholesterol": [cholesterol],
                "gluc": [gluc],
                "smoke": [smoke],
                "alco": [alco],
                "active": [active],
                "BMI": [BMI],
                "bp_ratio": [bp_ratio],
                "age_years": [age_years]
            })
            
            # Scaling dei dati
            input_data_scaled = input_data.copy()
            input_data_scaled[numerical_features] = scaler.transform(input_data[numerical_features])
            
            # Reordinamento delle feature per corrispondere al modello
            input_data_scaled = input_data_scaled[model_features]
            
            # Debugging: Verifica le colonne
            st.write("### Colonne dei Dati di Addestramento Scalati")
            st.write(training_data_scaled.columns.tolist())
            
            st.write("### Colonne dei Dati di Input Scalati")
            st.write(input_data_scaled.columns.tolist())
            
            # Predizione della classe
            try:
                prediction = model.predict(input_data_scaled)[0]  # Ottiene la classe predetta
                prediction_proba = model.predict_proba(input_data_scaled)[0]
            except Exception as e:
                st.error(f"Errore durante la predizione: {e}")
                st.stop()
            
            # Visualizzazione del risultato
            st.markdown("### 📈 Risultato della Predizione")
            if prediction == 1:
                st.error("⚠️ **È probabile che tu abbia problemi cardiaci.** Si consiglia di consultare un medico.")
            else:
                st.success("✅ **Non ci sono evidenze di problemi cardiaci.** Continua a mantenere uno stile di vita sano.")
            
            # Spiegazione con LIME
            if explainer is not None:
                try:
                    explanation = explainer.explain_instance(
                        data_row=input_data_scaled.values[0],
                        predict_fn=model.predict_proba,
                        num_features=len(model_features)
                    )
                    
                    # Visualizzazione delle spiegazioni
                    st.markdown("### 📊 Spiegazione della Predizione")
                    fig = explanation.as_pyplot_figure()
                    st.pyplot(fig)
                    
                    # Ordina le feature per importanza
                    sorted_features = sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)
                    top_features = [feature for feature, weight in sorted_features[:3]]
                    
                    # Genera suggerimenti basati sulle feature più influenti
                    st.markdown("### 📝 Suggerimenti per Migliorare")
                    generate_suggestions(top_features)
                except Exception as e:
                    st.error(f"Errore durante la generazione della spiegazione: {e}")
            else:
                st.warning("Non sono disponibili dati di addestramento per generare spiegazioni.")
        else:
            st.error("Assicurati che la pressione sistolica sia maggiore della diastolica.")
    else:
        st.error("Il modello o lo scaler non sono disponibili per effettuare la previsione.")
