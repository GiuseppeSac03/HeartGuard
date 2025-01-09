import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

def main():
    try:
        # Costruzione del percorso relativo al modello e allo scaler
        script_path = Path(__file__).resolve().parent
        model_path = script_path / 'svm_cardio_model.pkl'
        scaler_path = script_path / 'scaler.pkl'
        dataset_path = script_path / 'prepared_cardio_train.csv'

        # Verifica se i file esistono
        if not model_path.exists():
            raise FileNotFoundError(f"Il file del modello non esiste nel percorso {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Il file dello scaler non esiste nel percorso {scaler_path}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Il file del dataset non esiste nel percorso {dataset_path}")

        # Caricamento del modello e dello scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        print("Mean dello scaler caricato:", scaler.mean_)
        print("Varianza dello scaler caricato:", scaler.var_)

        # Definizione delle feature numeriche
        numerical_features = ['age_years', 'BMI', 'ap_hi', 'ap_lo', 'bp_ratio']

        # Verifica delle feature del modello
        if not hasattr(model, 'feature_names_in_'):
            raise AttributeError("Il modello caricato non ha l'attributo 'feature_names_in_'.")

        model_features = model.feature_names_in_
        print("Feature attese dal modello:", list(model_features))

        # Lista di diversi input per il test
        test_inputs = [
            {
                "gender": [1],
                "ap_hi": [120],
                "ap_lo": [80],
                "cholesterol": [1],
                "gluc": [1],
                "smoke": [0],
                "alco": [0],
                "active": [1],
                "BMI": [24.22],
                "bp_ratio": [1.5],
                "age_years": [40]
            },
            {
                "gender": [0],
                "ap_hi": [130],
                "ap_lo": [85],
                "cholesterol": [2],
                "gluc": [1],
                "smoke": [1],
                "alco": [0],
                "active": [0],
                "BMI": [30.5],
                "bp_ratio": [1.7],
                "age_years": [55]
            },
            {
                "gender": [1],
                "ap_hi": [110],
                "ap_lo": [70],
                "cholesterol": [1],
                "gluc": [1],
                "smoke": [0],
                "alco": [1],
                "active": [1],
                "BMI": [22.0],
                "bp_ratio": [1.6],
                "age_years": [35]
            }
            # Aggiungi altri dizionari con valori diversi se necessario
        ]

        # Funzione per eseguire la predizione su un singolo input
        def predict_single_input(input_dict, test_num):
            input_data = pd.DataFrame(input_dict)
            
            # Stampa dei valori originali
            print(f"\nTest {test_num} - Valori Originali:")
            print(input_data)
            
            # Scaling
            input_data_scaled = input_data.copy()
            input_data_scaled[numerical_features] = scaler.transform(input_data[numerical_features])
            
            # Stampa dei valori scalati
            print(f"Test {test_num} - Valori Scalati:")
            print(input_data_scaled)
            
            # Reordinamento delle feature secondo il modello
            input_data_scaled = input_data_scaled[model_features]
            
            # Stampa delle feature
            print(f"Test {test_num}:")
            print("Feature attese dal modello:", list(model_features))
            print("Feature fornite:", input_data_scaled.columns.tolist())
            
            # Predizione
            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(input_data_scaled)[:, 1]
                probability = prediction_proba[0] * 100
                print(f"La probabilità di avere problemi cardiaci è del {probability:.2f}%")
            else:
                print("Il modello non supporta 'predict_proba'.")
            
            # Funzione di decisione
            if hasattr(model, "decision_function"):
                decision = model.decision_function(input_data_scaled)
                print(f"Test {test_num} - Valore della funzione di decisione:", decision[0])
            else:
                print("Il modello non supporta 'decision_function'.")

        # Iterazione sui test
        for idx, data in enumerate(test_inputs, start=1):
            predict_single_input(data, idx)

        # Caricamento del dataset originale
        data = pd.read_csv(dataset_path)

        # Separazione delle feature e del target
        X = data.drop('cardio', axis=1)
        y = data['cardio']

        # Divisione in training e test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      

        # Predizione sui dati di test
        y_pred = model.predict(X_test)

        # Predizione delle probabilità se disponibile
        if hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        elif hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        # Calcolo delle metriche
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = "Non disponibile"

        print("\n--- Valutazione del Modello sul Test Set ---")
        print(f"Accuratezza: {accuracy:.4f}")
        print(f"Precisione: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")

    except Exception as e:
        print(f"Si è verificato un errore: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
