# **Analisi e Selezione delle Feature per il Dataset Cardio**

## **1. Introduzione**
Abbiamo analizzato e preparato il dataset per il task di predizione delle malattie cardiovascolari. Il processo ha incluso la selezione delle feature più rilevanti tramite diversi metodi e la preparazione del dataset per l'uso nei modelli di machine learning.

---

## **2. Analisi delle Feature**
### **Random Forest Importance**
La Random Forest calcola l'importanza delle feature in base al decremento dell'impurità (Gini). Risultati principali:
- **Feature più importanti**:
  - **BMI** (0.4746)
  - **age_years** (0.1553)
  - **ap_hi** (0.1326)
- **Feature meno influenti**:
  - **smoke** (0.0099)
  - **alco** (0.0083)

### **Permutation Importance**
Questo metodo mescola casualmente una feature e valuta l'impatto sulla performance del modello. Risultati principali:
- **Feature più importanti**:
  - **age_years** (0.2781)
  - **ap_hi** (0.2726)
  - **BMI** (0.2687)
- **Miglioramenti per smoke/alco**:
  - **smoke** (0.0335)
  - **alco** (0.0197)

**Conclusione**: La Permutation Importance evidenzia un impatto leggermente maggiore per le feature con valori meno frequenti (es. **smoke** e **alco**), riducendo la distorsione causata dagli squilibri.

---

## **3. Selezione delle Feature**
Dopo le analisi, abbiamo scelto le seguenti feature per il dataset finale:
1. **age_years**: Età in anni.
2. **BMI**: Indice di Massa Corporea.
3. **ap_hi**: Pressione arteriosa sistolica.
4. **ap_lo**: Pressione arteriosa diastolica.
5. **bp_ratio**: Rapporto tra ap_hi e ap_lo.
6. **cholesterol**: Livello di colesterolo (1: normale, 2: sopra il normale, 3: molto sopra il normale).
7. **gluc**: Livello di glucosio (1: normale, 2: sopra il normale, 3: molto sopra il normale).
8. **gender**: Genere (1: Donna, 2: Uomo).
9. **active**: Attività fisica (0: No, 1: Sì).
10. **smoke**: Fumatore (0: No, 1: Sì).
11. **alco**: Consumo di alcol (0: No, 1: Sì).

**Feature rimosse**:
- **id**: Non rilevante.
- **age** (in giorni): Ridondante rispetto a **age_years**.
- **weight** e **height**: Già rappresentate dal **BMI**.