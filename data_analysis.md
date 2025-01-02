# **Documento di Analisi dei Dati**

## **1. Obiettivi**
L'obiettivo di questa analisi è identificare, collezionare ed esplorare i dati necessari per predire il rischio di malattie cardiache e identificare i fattori principali che influenzano questa condizione.

---

## **2. Dataset**
### **Fonte**
- Nome: **Heart Disease Dataset**
- Fonte: [Kaggle](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)

### **Caratteristiche del Dataset**
- **Formato**: CSV
- **Numero di record**: 10,000
- **Numero di colonne**: 21
- **Significato delle colonne**:
  - **Age**: Età dell'individuo.
  - **Gender**: Genere (Maschio o Femmina).
  - **Blood Pressure**: Pressione sanguigna (sistolica).
  - **Cholesterol Level**: Livello di colesterolo totale.
  - **Exercise Habits**: Abitudini di esercizio fisico (Basso, Medio, Alto).
  - **Smoking**: Fumatore (Sì o No).
  - **Family Heart Disease**: Storia familiare di malattie cardiache (Sì o No).
  - **Diabetes**: Presenza di diabete (Sì o No).
  - **BMI**: Indice di massa corporea.
  - **High Blood Pressure**: Pressione alta (Sì o No).
  - **Low HDL Cholesterol**: HDL basso (Sì o No).
  - **High LDL Cholesterol**: LDL alto (Sì o No).
  - **Alcohol Consumption**: Livello di consumo di alcol (Nessuno, Basso, Medio, Alto).
  - **Stress Level**: Livello di stress (Basso, Medio, Alto).
  - **Sleep Hours**: Ore di sonno giornaliere.
  - **Sugar Consumption**: Livello di consumo di zucchero (Basso, Medio, Alto).
  - **Triglyceride Level**: Livello di trigliceridi.
  - **Fasting Blood Sugar**: Livello di zucchero a digiuno.
  - **CRP Level**: Livello di proteina C-reattiva (marker di infiammazione).
  - **Homocysteine Level**: Livello di omocisteina.
  - **Heart Disease Status**: Presenza di malattie cardiache (Sì o No).

---

## **3. Esplorazione dei Dati**
### **Statistiche di Base**
- **Dimensione del dataset**: 10,000 record, 21 colonne.
- **Formato**: CSV con dati numerici e categorici.

### **Distribuzione della Variabile Target**
- **Heart Disease Status**:
  - `0` (Nessuna malattia cardiaca): **80%**
  - `1` (Malattia cardiaca): **20%**

### **Considerazioni sulla distribuzione**
1. **Sbilanciamento**:
   - La distribuzione è altamente sbilanciata, con la classe `0` (80%) dominante rispetto alla classe `1` (20%).
   - Questo può portare a problemi nei modelli di Machine Learning, come la tendenza a favorire la classe dominante.
2. **Implicazioni per l'analisi**:
   - Metriche come l'accuratezza potrebbero non essere affidabili per valutare il modello.
   - Sarà necessario utilizzare strategie per gestire lo sbilanciamento dei dati:
     - Oversampling della classe `1`.
     - Undersampling della classe `0`.
     - Uso di pesi di classe o metriche appropriate come F1-Score o AUC-ROC.

---

## **4. Feature Engineering**
Durante l'analisi, sono state create due nuove feature per catturare relazioni complesse nei dati:

1. **Stress_Sleep_Index**:
   - **Definizione**: Combina il livello di stress (`Stress Level`) e le ore di sonno (`Sleep Hours`) in un unico indicatore.
   - **Motivazione**:
     - Studi clinici mostrano che lo stress e il sonno hanno un impatto combinato sulla salute cardiovascolare.
     - Lo `Stress_Sleep_Index` permette di rappresentare questa relazione complessa.
   - **Contributo**: Ha mostrato un'importanza predittiva alta (**8.11%**) ed è moderatamente correlata a `Sleep Hours` (-0.52), suggerendo che entrambe le variabili aggiungono valore.

2. **Cholesterol_BMI_Ratio**:
   - **Definizione**: Rapporto tra i livelli di colesterolo totale (`Cholesterol Level`) e l'indice di massa corporea (`BMI`).
   - **Motivazione**:
     - Il colesterolo e il BMI sono indicatori metabolici chiave. Il loro rapporto rappresenta una misura combinata dello stato metabolico.
   - **Contributo**: Ha mostrato un'importanza predittiva significativa (**7.75%**), superiore a quella di `Cholesterol Level` considerato singolarmente.

---

## **5. Feature Selection**
### **Variabili Selezionate per il Modello**
Le seguenti feature sono state identificate come le più rilevanti sulla base della feature importance calcolata con Random Forest:

1. **CRP Level**: 8.23%
2. **Homocysteine Level**: 8.19%
3. **Sleep Hours**: 8.12%
4. **Stress_Sleep_Index**: 8.11%
5. **BMI**: 8.05%
6. **Cholesterol_BMI_Ratio**: 7.75%
7. **Triglyceride Level**: 7.65%
8. **Cholesterol Level**: 7.08%
9. **Fasting Blood Sugar**: 6.98%
10. **Age**: 6.77%

### **Variabili Escluse**
Le variabili con importanza predittiva inferiore all'1% sono state escluse per semplificare il modello:
- **Stress Level**: 1.33%
- **Smoking_1.0**, **High LDL Cholesterol**, ecc.: <1%.

---

## **6. Qualità dei Dati**
### **Problemi Identificati**
1. **Valori mancanti**:
   - Colonne con valori mancanti significativi:
     - `Alcohol Consumption` (25.86% di dati mancanti), risolto creando una categoria `Unknown`.
2. **Formati misti**:
   - Alcune colonne categoriche (`Gender`, `Exercise Habits`, ecc.) sono state codificate in valori numerici.
3. **Distribuzione sbilanciata**:
   - La variabile target (`Heart Disease Status`) è altamente sbilanciata, richiedendo tecniche di bilanciamento.

---

## **7. Output**
Il dataset finale contiene 26 colonne (inclusi le feature derivate) e 10,000 record. Saranno utilizzate le seguenti variabili per il modello di Machine Learning:
- **CRP Level**, **Homocysteine Level**, **Sleep Hours**, **Stress_Sleep_Index**, **BMI**, **Cholesterol_BMI_Ratio**, **Triglyceride Level**, **Cholesterol Level**, **Fasting Blood Sugar**, e **Age**.

Questa selezione rappresenta le variabili più predittive, bilanciando complessità e performance.

---
