# HeartGuard: Prevenzione Cardiaca Intelligente

HeartGuard è un'applicazione basata su Machine Learning progettata per prevedere il rischio di malattie cardiache e fornire raccomandazioni personalizzate per migliorare la salute del cuore.

## Dataset
Questo progetto utilizza il [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data) da Kaggle, che include fattori chiave come età, livelli di colesterolo, abitudini di esercizio fisico e altro.

## Struttura della Repository

### Notebook
1. **`data_analysis.ipynb`**  
   Esplora il dataset per identificare le variabili principali che influenzano il rischio cardiovascolare. Include la gestione degli outlier e l'analisi delle correlazioni.

2. **`data_preparation.ipynb`**  
   Prepara il dataset per la modellazione creando nuove feature (es. BMI, rapporto pressione arteriosa), scalando i valori e selezionando le feature più rilevanti.

3. **`modelling.ipynb`**  
   Implementa vari modelli di Machine Learning (Regressione Logistica, Random Forest, SVM, KNN) per prevedere le malattie cardiache, seleziona il modello migliore e ottimizza i parametri.

4. **`evaluation.ipynb`**  
   Valuta il modello finale utilizzando metriche come matrice di confusione, curva ROC-AUC e SHAP per l'interpretabilità.

### Applicazione
5. **`app.py`**  
   Applicazione web basata su Streamlit in cui gli utenti possono inserire i propri dati di salute per ricevere una previsione del rischio e raccomandazioni personalizzate.

### Dati
6. **`cardio_train.csv`**  
   Il dataset originale contenente i dati grezzi sulla salute cardiovascolare.

7. **`prepared_cardio_train.csv`**  
   Il dataset processato con feature selezionate e valori normalizzati.

8. **`X_train`**  
   Dataset di training utilizzato per l'analisi SHAP per spiegare le predizioni del modello.

### Modelli e Utility
9. **`scaler.pkl`**  
   Scaler pre-addestrato per normalizzare nuovi input durante la previsione.

10. **`svm_cardio_model.pkl`**  
   Modello finale SVM ottimizzato per prevedere il rischio di malattie cardiovascolari.

### Media
11. **`imgs/`**  
    Contiene il logo.

## Funzionalità
- Predice la probabilità di malattie cardiache basandosi sui dati degli utenti.
- Fornisce raccomandazioni personalizzate per ridurre il rischio cardiovascolare.
- Offre spiegazioni basate su SHAP per aumentare la fiducia e la comprensione dell'utente.

---

### Disclaimer
HeartGuard è destinato esclusivamente a scopi educativi e non deve essere utilizzato come sostituto del parere medico professionale.
