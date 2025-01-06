# Documento di Analisi dei Dati

## 1. Obiettivi

L'obiettivo di questa analisi è esplorare e preparare un dataset per la previsione della presenza di malattie cardiovascolari. Attraverso l'esplorazione dei dati, si cercherà di identificare i fattori principali che influenzano questa condizione.

---

## 2. Dataset

### Fonte

- **Nome**: Cardiovascular Disease Dataset  
- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)

### Caratteristiche del Dataset

- **Formato**: CSV  
- **Numero di record**: 70,000  
- **Numero di colonne**: 12  

### Significato delle colonne:

- **age**: Età in giorni.  
- **height**: Altezza (in cm).  
- **weight**: Peso (in kg).  
- **gender**: Genere (1: Donna, 2: Uomo).  
- **ap_hi**: Pressione arteriosa sistolica.  
- **ap_lo**: Pressione arteriosa diastolica.  
- **cholesterol**: Livello di colesterolo (1: normale, 2: sopra il normale, 3: molto sopra il normale).  
- **gluc**: Livello di glucosio (1: normale, 2: sopra il normale, 3: molto sopra il normale).  
- **smoke**: Fumatore (0: No, 1: Sì).  
- **alco**: Consumo di alcol (0: No, 1: Sì).  
- **active**: Attività fisica (0: No, 1: Sì).  
- **cardio**: Presenza di malattia cardiovascolare (0: No, 1: Sì).  

### Valori Mancanti

Nessun valore mancante è stato rilevato nel dataset:

| Colonna        | Valori Mancanti |
|----------------|-----------------|
| age            | 0               |
| gender         | 0               |
| height         | 0               |
| weight         | 0               |
| ap_hi          | 0               |
| ap_lo          | 0               |
| cholesterol    | 0               |
| gluc           | 0               |
| smoke          | 0               |
| alco           | 0               |
| cardio         | 0               |

---

## 3. Esplorazione dei Dati

### Distribuzione della Variabile Target

- **cardio**:
  - 0 (Nessuna malattia cardiovascolare): ~50%  
  - 1 (Malattia cardiovascolare): ~50%  

La distribuzione è bilanciata, rendendo affidabili metriche standard come l'accuracy.

### Distribuzione delle Variabili Principali

- **age** (convertita in anni):  
  - Range: 30-65 anni.  
  - Distribuzione concentrata tra i 50 e i 60 anni, con un picco intorno ai 55 anni.  

- **height** e **weight**:  
  - **height**: Alcuni valori outlier sotto i 120 cm o sopra i 200 cm.  
  - **weight**: Outlier significativi sopra i 150 kg.  

- **ap_hi** e **ap_lo**:  
  - Evidenti outlier con valori estremamente alti (es. >2000 per ap_hi).  
  - Richiedono una verifica accurata e potenziale correzione.  

- **cholesterol** e **gluc**:  
  - Valori maggiormente concentrati nella categoria 1 (normale), con un minor numero di casi nelle categorie 2 e 3.  

### Matrice di Correlazione

- **age**, **weight**, e **cholesterol** mostrano una correlazione positiva con **cardio**.  
- Bassa correlazione tra altre variabili come **height** o **ap_lo** con **cardio**.

### Outlier Identificati

- **ap_hi** e **ap_lo** presentano valori anomali elevati (es. >10,000).  
- **height** e **weight** hanno valori estremi da analizzare e potenzialmente rimuovere o trasformare.  

---
