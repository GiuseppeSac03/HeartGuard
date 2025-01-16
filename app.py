import streamlit as st
import pandas as pd
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import shap  # <--- Import SHAP

# -----------------------------
# Configurazione della pagina
# -----------------------------
st.set_page_config(
    page_title="Predictor di Problemi Cardiaci",
    page_icon="❤️", 
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Funzioni di utilità
# -----------------------------
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

def calculate_bp_ratio(ap_hi, ap_lo):
    if ap_lo == 0:
        return 0
    return round(ap_hi / ap_lo, 2)

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

# Caricamento dei dati di addestramento scalati
@st.cache_resource
def load_training_data_scaled():
    try:
        training_data_scaled = pd.read_csv('X_train.csv')
        return training_data_scaled
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati di addestramento scalati: {e}")
        return None

# Funzione per generare suggerimenti basati sulle feature
def generate_suggestions(top_features):
    for feature in top_features:
        if feature.startswith("BMI"):
            st.write("- **BMI:** Considera di mantenere o raggiungere un BMI nella fascia normale (18.5 - 24.9) attraverso una dieta equilibrata e attività fisica regolare.")
        elif feature.startswith("bp_ratio"):
            st.write("- **Rapporto BP:** Mantieni la pressione arteriosa sotto controllo seguendo le indicazioni mediche, riducendo l'assunzione di sale e gestendo lo stress.")
        elif feature.startswith("ap_hi"):
            st.write("- **Pressione Arteriosa Sistolica:** Mantienila sotto controllo seguendo le indicazioni mediche, riducendo l'assunzione di sale e gestendo lo stress.")
        elif feature.startswith("ap_lo"):
            st.write("- **Pressione Arteriosa Diastolica:** Mantienila sotto controllo seguendo le indicazioni mediche, riducendo l'assunzione di sale e gestendo lo stress.")
        elif feature.startswith("age_years"):
            st.write("- **Età:** L'età è un fattore non modificabile, ma puoi concentrarti sugli altri aspetti per ridurre il rischio.")
        elif feature.startswith("gender"):
            st.write("- **Genere:** Alcuni fattori di rischio variano in base al genere. Consulta un medico per consigli personalizzati.")
        elif feature.startswith("cholesterol"):
            st.write("- **Colesterolo:** Monitora e gestisci i livelli di colesterolo con una dieta sana e, se necessario, farmaci prescritti dal medico.")
        elif feature.startswith("gluc"):
            st.write("- **Glucosio:** Mantieni i livelli di glucosio sotto controllo con dieta equilibrata e attività fisica.")
        elif feature.startswith("smoke"):
            st.write("- **Fumo:** Se fumi, considera di smettere per ridurre significativamente il rischio di problemi cardiaci.")
        elif feature.startswith("alco"):
            st.write("- **Consumo di Alcol:** Limita il consumo di alcol per migliorare la salute del cuore.")
        elif feature.startswith("active"):
            st.write("- **Attività Fisica:** Aumenta il livello di attività fisica per migliorare la salute cardiovascolare.")

# -----------------------------
# Inizializzazione di session_state
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "age_years" not in st.session_state:
    st.session_state.age_years = 40
if "gender" not in st.session_state:
    st.session_state.gender = 1  # 1 = donna, 2 = uomo
if "height" not in st.session_state:
    st.session_state.height = 170
if "weight" not in st.session_state:
    st.session_state.weight = 70
if "ap_hi" not in st.session_state:
    st.session_state.ap_hi = 120
if "ap_lo" not in st.session_state:
    st.session_state.ap_lo = 80
if "cholesterol" not in st.session_state:
    st.session_state.cholesterol = 1
if "gluc" not in st.session_state:
    st.session_state.gluc = 1
if "smoke" not in st.session_state:
    st.session_state.smoke = 0
if "alco" not in st.session_state:
    st.session_state.alco = 0
if "active" not in st.session_state:
    st.session_state.active = 0

# Questi tengono traccia della previsione
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_proba" not in st.session_state:
    st.session_state.prediction_proba = None

# IMPORTANTE: salveremo qui il DataFrame scaled
if "input_data_scaled" not in st.session_state:
    st.session_state.input_data_scaled = None

# -----------------------------
# Pulsanti di navigazione
# -----------------------------
def go_next_step():
    st.session_state.step += 1

def go_prev_step():
    st.session_state.step -= 1
    st.session_state.prediction_done = False

# -----------------------------
# Caricamento di modello e scaler
# -----------------------------
model = load_model()
scaler = load_scaler()
training_data_scaled = load_training_data_scaled()

numerical_features = ['age_years', 'BMI', 'ap_hi', 'ap_lo', 'bp_ratio']
model_features = ['gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                  'smoke', 'alco', 'active', 'BMI', 'bp_ratio', 'age_years']


# Per SHAP: Creiamo un mini-background
explainer_shap = None
if training_data_scaled is not None and model is not None:
    background_size = 20
    if len(training_data_scaled) > background_size:
        background_sample = training_data_scaled.sample(background_size, random_state=42)
    else:
        background_sample = training_data_scaled

    try:
        explainer_shap = shap.Explainer(
            lambda x: model.predict_proba(x)[:, 1],
            background_sample,
            feature_names=model_features
        )
    except Exception as e:
        st.write(f"SHAP Explainer non creato: {e}")

# -----------------------------
# Titolo Principale
# -----------------------------
st.title("❤️ Predictor di Problemi Cardiaci")
st.markdown("""
Questa applicazione prevede la presenza di problemi cardiaci in base ai dati inseriti.
Compila i seguenti passi uno per volta.
""")
st.image(
    "imgs/logo_scritta.png",
    caption="Prenditi cura del tuo cuore!",
    use_container_width=True
)

# -----------------------------
# STEP-BY-STEP: Visualizza lo step corrente
# -----------------------------
if st.session_state.step == 1:
    st.subheader("Step 1: Età")
    st.session_state.age_years = st.number_input(
        "🧓 Inserisci la tua età (in anni)",
        min_value=0, max_value=120, value=st.session_state.age_years, step=1
    )

    # Pulsanti centrati
    col = st.columns([2,1,1,2])  # [Spazio, Bottone, Bottone, Spazio]
    with col[1]:
        st.write("")  # Lasciamo vuoto
    with col[2]:
        st.button("Avanti", on_click=go_next_step, type="primary")

elif st.session_state.step == 2:
    st.subheader("Step 2: Genere")
    gender_str = st.selectbox(
        "⚧ Seleziona il tuo genere",
        ["Donna", "Uomo"],
        index=0 if st.session_state.gender == 1 else 1
    )
    st.session_state.gender = 1 if gender_str.startswith("Donna") else 2

    col = st.columns([2,1,1,2])
    with col[1]:
        st.button("Indietro", on_click=go_prev_step)
    with col[2]:
        st.button("Avanti", on_click=go_next_step, type="primary")


elif st.session_state.step == 3:
    st.subheader("Step 3: Altezza e Peso")
    st.session_state.height = st.number_input(
        "📏 Altezza (in cm)",
        min_value=50, max_value=250, value=st.session_state.height, step=1
    )
    st.session_state.weight = st.number_input(
        "⚖️ Peso (in kg)",
        min_value=10, max_value=300, value=st.session_state.weight, step=1
    )

    col = st.columns([2,1,1,2])
    with col[1]:
        st.button("Indietro", on_click=go_prev_step)
    with col[2]:
        st.button("Avanti", on_click=go_next_step, type="primary")

elif st.session_state.step == 4:
    st.subheader("Step 4: Pressione Arteriosa")
    st.session_state.ap_hi = st.number_input(
        "Sistolica (ap_hi)",
        min_value=50, max_value=250, value=st.session_state.ap_hi, step=1
    )
    st.session_state.ap_lo = st.number_input(
        "Diastolica (ap_lo)",
        min_value=30, max_value=150, value=st.session_state.ap_lo, step=1
    )

    if st.session_state.ap_hi <= st.session_state.ap_lo:
        st.error("La pressione sistolica deve essere maggiore di quella diastolica.")

    col = st.columns([2,1,1,2])
    with col[1]:
        st.button("Indietro", on_click=go_prev_step)
    with col[2]:
        st.button("Avanti", on_click=go_next_step, type="primary",
                  disabled=(st.session_state.ap_hi <= st.session_state.ap_lo))

elif st.session_state.step == 5:
    st.subheader("Step 5: Colesterolo e Glucosio")
    
    # Livello di Colesterolo
    chol_str = st.selectbox(
        "🔬 Livello di Colesterolo",
        ["Normale", "Sopra il normale", "Molto sopra il normale"],  
        index=st.session_state.cholesterol - 1
    )
    st.session_state.cholesterol = ["Normale", "Sopra il normale", "Molto sopra il normale"].index(chol_str) + 1

    # Livello di Glucosio
    gluc_str = st.selectbox(
        "🍭 Livello di Glucosio",
        ["Normale", "Sopra il normale", "Molto sopra il normale"],
        index=st.session_state.gluc - 1
    )
    st.session_state.gluc = ["Normale", "Sopra il normale", "Molto sopra il normale"].index(gluc_str) + 1

    col = st.columns([2,1,1,2])
    with col[1]:
        st.button("Indietro", on_click=go_prev_step)
    with col[2]:
        st.button("Avanti", on_click=go_next_step, type="primary")


elif st.session_state.step == 6:
    st.subheader("Step 6: Abitudini di Vita")
    
    # Fumatore
    smoke_str = st.selectbox(
        "Sei un fumatore?",
        ["No", "Sì"], 
        index=st.session_state.smoke
    )
    st.session_state.smoke = ["No", "Sì"].index(smoke_str)

    # Consumo di alcol
    alco_str = st.selectbox(
        "Consumi alcol?",
        ["No", "Sì"],  
        index=st.session_state.alco
    )
    st.session_state.alco = ["No", "Sì"].index(alco_str)

    # Attività fisica
    active_str = st.selectbox(
        "Sei attivo fisicamente?",
        ["No", "Sì"],  
        index=st.session_state.active
    )
    st.session_state.active = ["No", "Sì"].index(active_str)

    col = st.columns([2,1,1,2])
    with col[1]:
        st.button("Indietro", on_click=go_prev_step)
    with col[2]:
        st.button("Avanti", on_click=go_next_step, type="primary")

elif st.session_state.step == 7:
    st.subheader("Step 7: Riepilogo e Previsione")
    
    BMI = calculate_bmi(st.session_state.weight, st.session_state.height)
    bp_ratio = calculate_bp_ratio(st.session_state.ap_hi, st.session_state.ap_lo)

    # Riepilogo con card visive
    st.markdown("### Dettagli Personali")
    st.write(f"- **Genere:** {'Donna' if st.session_state.gender == 1 else 'Uomo'}")
    st.write(f"- **Età:** {st.session_state.age_years} anni")
    st.write(f"- **Altezza:** {st.session_state.height} cm")
    st.write(f"- **Peso:** {st.session_state.weight} kg")
    st.write(f"- **BMI:** {BMI:.2f}")

    st.markdown("### Parametri Vitali")
    st.write(f"- **Pressione Sistolica (ap_hi):** {st.session_state.ap_hi}")
    st.write(f"- **Pressione Diastolica (ap_lo):** {st.session_state.ap_lo}")
    st.write(f"- **Rapporto BP:** {bp_ratio:.2f}")
    st.write(f"- **Colesterolo:** {['Normale', 'Sopra il normale', 'Molto sopra il normale'][st.session_state.cholesterol - 1]}")
    st.write(f"- **Glucosio:** {['Normale', 'Sopra il normale', 'Molto sopra il normale'][st.session_state.gluc - 1]}")

    st.markdown("### Abitudini di Vita")
    st.write(f"- **Fumatore:** {'Sì' if st.session_state.smoke == 1 else 'No'}")
    st.write(f"- **Consumo Alcol:** {'Sì' if st.session_state.alco == 1 else 'No'}")
    st.write(f"- **Attività Fisica:** {'Sì' if st.session_state.active == 1 else 'No'}")

    # Pulsanti centrati: "Indietro" e "Calcola Previsione"
    col = st.columns([2,1,1,2])
    with col[1]:
        st.button("Indietro", on_click=go_prev_step)

    if model is not None and scaler is not None:
        if st.session_state.ap_hi > st.session_state.ap_lo:
            with col[2]:
                if st.button("Calcola Previsione", type="primary"):
                    # Prepariamo il DataFrame
                    input_data = pd.DataFrame({
                        "gender": [st.session_state.gender],
                        "ap_hi": [st.session_state.ap_hi],
                        "ap_lo": [st.session_state.ap_lo],
                        "cholesterol": [st.session_state.cholesterol],
                        "gluc": [st.session_state.gluc],
                        "smoke": [st.session_state.smoke],
                        "alco": [st.session_state.alco],
                        "active": [st.session_state.active],
                        "BMI": [BMI],
                        "bp_ratio": [bp_ratio],
                        "age_years": [st.session_state.age_years]
                    })

                    # Scalatura
                    input_data_scaled = input_data.copy()
                    input_data_scaled[numerical_features] = scaler.transform(input_data[numerical_features])
                    input_data_scaled = input_data_scaled[model_features]

                    # Salviamo in session_state
                    st.session_state.input_data_scaled = input_data_scaled

                    try:
                        st.session_state.prediction_result = model.predict(input_data_scaled)[0]
                        st.session_state.prediction_proba = model.predict_proba(input_data_scaled)[0]
                        st.session_state.prediction_done = True
                    except Exception as e:
                        st.error(f"Errore durante la predizione: {e}")
        else:
            st.error("Assicurati che la pressione sistolica sia maggiore della diastolica.")
    else:
        st.warning("Il modello o lo scaler non sono caricati correttamente, impossibile calcolare la previsione.")

    # Mostriamo il risultato
    if st.session_state.prediction_done:
        st.markdown("### Risultato della Predizione")
        if st.session_state.prediction_result == 1:
            st.error("⚠️ **È probabile che tu abbia problemi cardiaci.** Consigliamo di consultare un medico.")
        else:
            st.success("✅ Non ci sono evidenze di probabili problemi cardiaci. Continua a mantenere uno stile di vita sano.")

        # Spiegazione SHAP
        if explainer_shap is not None:
            try:
                if st.button("Calcola Spiegazione SHAP"):
                    if st.session_state.input_data_scaled is None:
                        st.error("Prima calcola la previsione per generare i dati scalati!")
                    else:
                        with st.spinner("Calcolo della spiegazione SHAP in corso..."):
                            shap_values = explainer_shap(st.session_state.input_data_scaled)

                        st.markdown("### Spiegazione con SHAP")
                        ax = shap.plots.waterfall(shap_values[0], show=False)
                        fig = ax.figure
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Errore durante la generazione della spiegazione SHAP: {e}")
        else:
            st.warning("Spiegazione SHAP non disponibile per questo modello.")
