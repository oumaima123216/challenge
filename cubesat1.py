# ============================================================================#
# 🚀 SYSTÈME COMPLET : LSTM + ISOLATION FOREST
# Monitoring autonome batterie CubeSat
# Colle ce script dans Google Colab
# ============================================================================#
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib
import os

# Reproductibilité (dans la mesure du possible)
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------------------------------------------#
# PARTIE 1 : GÉNÉRATION DE DONNÉES RÉALISTES
# -----------------------------------------------------------------------------#
def generer_donnees_batterie_realistes(nb_orbites=480):
    """
    Génère des données de batterie CubeSat réalistes.
    nb_orbites : nombre d'orbites simulées (ex: 480 ~ 30 jours si 16 orbites/jour)
    """
    VOLTAGE_NOMINAL = 7.4
    VOLTAGE_MIN = 6.0
    VOLTAGE_MAX = 8.4

    donnees = []
    for orbite in range(nb_orbites):
        # Phase Eclipse (35 min)
        duree_eclipse = 35
        for t in range(duree_eclipse):
            progression = t / duree_eclipse
            soc = 80 - (progression * 25)
            voltage = VOLTAGE_NOMINAL - (progression * 0.8)
            courant = -0.5 - np.random.normal(0, 0.08)
            temperature = 10 - (progression * 15)
            voltage += np.random.normal(0, 0.02)
            temperature += np.random.normal(0, 1.5)

            anomalie = 0
            if np.random.random() < 0.01:
                scenario = np.random.choice(['deep_discharge', 'thermal_runaway', 'cell_imbalance'])
                if scenario == 'deep_discharge':
                    voltage -= 0.8
                    soc -= 20
                    anomalie = 1
                elif scenario == 'thermal_runaway':
                    temperature += 30
                    courant *= 2.5
                    anomalie = 1
                elif scenario == 'cell_imbalance':
                    voltage -= 0.4
                    anomalie = 1

            donnees.append({
                'orbite': orbite,
                'temps': t,
                'phase': 'eclipse',
                'voltage': max(VOLTAGE_MIN, voltage),
                'courant': courant,
                'temperature': temperature,
                'soc': max(0, soc),
                'anomalie': anomalie
            })

        # Phase Lumière (55 min)
        duree_lumiere = 55
        for t in range(duree_lumiere):
            progression = t / duree_lumiere
            soc = 55 + (progression * 30)
            voltage = VOLTAGE_NOMINAL + (progression * 0.6)
            courant = 0.8 + np.random.normal(0, 0.1)
            temperature = -5 + (progression * 30)
            voltage += np.random.normal(0, 0.02)
            temperature += np.random.normal(0, 1.5)

            anomalie = 0
            if np.random.random() < 0.005:
                scenario = np.random.choice(['overcharge', 'panel_degradation'])
                if scenario == 'overcharge':
                    voltage += 1.2
                    temperature += 25
                    anomalie = 1
                elif scenario == 'panel_degradation':
                    courant *= 0.2
                    anomalie = 1

            donnees.append({
                'orbite': orbite,
                'temps': t + duree_eclipse,
                'phase': 'lumiere',
                'voltage': min(VOLTAGE_MAX, voltage),
                'courant': courant,
                'temperature': temperature,
                'soc': min(100, soc),
                'anomalie': anomalie
            })

    df = pd.DataFrame(donnees)
    return df

# Génération
print("📊 Génération des données...")
df = generer_donnees_batterie_realistes(nb_orbites=480)
print(f"✅ {len(df)} points générés — anomalies totales : {df['anomalie'].sum()} ({df['anomalie'].sum()/len(df)*100:.3f}%)")
print(df.head())

# -----------------------------------------------------------------------------#
# PARTIE 2 : ISOLATION FOREST
# -----------------------------------------------------------------------------#
features_iforest = ['voltage', 'courant', 'temperature', 'soc']
X_iforest = df[features_iforest].values
y_iforest = df['anomalie'].values.astype(int)

# Split
X_train_if, X_test_if, y_train_if, y_test_if = train_test_split(
    X_iforest, y_iforest, test_size=0.2, random_state=42, stratify=y_iforest
)

# Entraîner seulement sur données normales (0)
X_train_normal = X_train_if[y_train_if == 0]
if len(X_train_normal) < 10:
    raise ValueError("Pas assez d'échantillons normaux pour entraîner IsolationForest.")

max_samples = min(256, len(X_train_normal))
model_iforest = IsolationForest(
    contamination=0.01,
    n_estimators=100,
    max_samples=max_samples,
    random_state=42,
    n_jobs=-1
)

print("\n🌲 Entraînement IsolationForest...")
start = time.time()
model_iforest.fit(X_train_normal)
dur = time.time() - start
print(f"✅ IForest entraîné en {dur:.2f}s")

# Évaluation
y_pred_if = model_iforest.predict(X_test_if)
y_pred_if = (y_pred_if == -1).astype(int)

accuracy_if = accuracy_score(y_test_if, y_pred_if)
precision_if = precision_score(y_test_if, y_pred_if, zero_division=0)
recall_if = recall_score(y_test_if, y_pred_if, zero_division=0)
print(f"IForest — Acc: {accuracy_if:.4f}, Prec: {precision_if:.4f}, Rec: {recall_if:.4f}")

# -----------------------------------------------------------------------------#
# PARTIE 3 : PRÉPARATION & ENTRAÎNEMENT LSTM
# -----------------------------------------------------------------------------#
def preparer_sequences_lstm(df, fenetre=10, features=['voltage','courant','temperature','soc']):
    X = df[features].values
    y = df['anomalie'].values.astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - fenetre):
        X_seq.append(X_scaled[i:i+fenetre])
        y_seq.append(y[i+fenetre])
    return np.array(X_seq), np.array(y_seq), scaler

print("\n🧠 Préparation des séquences LSTM...")
FENETRE = 10
X_lstm, y_lstm, scaler = preparer_sequences_lstm(df, fenetre=FENETRE)
print(f"Shapes LSTM: X={X_lstm.shape}, y={y_lstm.shape}")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X_lstm, y_lstm, test_size=0.3, random_state=42, stratify=y_lstm)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# Construire modèle LSTM léger
model_lstm = keras.Sequential([
    keras.layers.Input(shape=(X_lstm.shape[1], X_lstm.shape[2])),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(16),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

print("\n🔧 Résumé modèle LSTM :")
model_lstm.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

print("\n⏳ Entraînement LSTM...")
history = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
print("✅ LSTM entraîné")

# Évaluation LSTM
test_results = model_lstm.evaluate(X_test, y_test, verbose=0)
print(f"LSTM — Loss: {test_results[0]:.4f}, Acc: {test_results[1]:.4f}, Prec: {test_results[2]:.4f}, Rec: {test_results[3]:.4f}")

# -----------------------------------------------------------------------------#
# UTIL : Mesure de latence (warm-up + moyenne)
# -----------------------------------------------------------------------------#
def measure_latency_predict(predict_fn, sample_input, n_runs=100, warmup=10):
    # Warm-up
    for _ in range(warmup):
        _ = predict_fn(sample_input)
    start = time.time()
    for _ in range(n_runs):
        _ = predict_fn(sample_input)
    total = time.time() - start
    latency_ms = (total / n_runs) * 1000.0
    return latency_ms

# Mesurer latence IForest (entrée 2D)
sample_if = X_test_if[:1]
latence_if = measure_latency_predict(lambda x: model_iforest.predict(x), sample_if, n_runs=100, warmup=10)

# Mesurer latence LSTM (entrée 3D)
sample_lstm = X_test[:1]
latence_lstm = measure_latency_predict(lambda x: model_lstm.predict(x, verbose=0), sample_lstm, n_runs=100, warmup=10)

print(f"\n⏱️ Latence (moyenne) — IForest: {latence_if:.3f} ms / préd , LSTM: {latence_lstm:.3f} ms / préd")

# -----------------------------------------------------------------------------#
# PARTIE 4 : SYSTÈME HYBRIDE (fusion)
# -----------------------------------------------------------------------------#
class SystemeHybride:
    def __init__(self, model_iforest, model_lstm, scaler, buffer_size=FENETRE):
        self.model_iforest = model_iforest
        self.model_lstm = model_lstm
        self.scaler = scaler
        self.buffer = []
        self.buffer_size = buffer_size
        self.seuil_lstm = 0.5
        self.seuil_fusion = 0.6

    def analyser(self, voltage, courant, temperature, soc):
        """
        Analyse une mesure avec les 2 modèles.
        Retourne scores et décision.
        """
        donnees = np.array([[voltage, courant, temperature, soc]])
        # IsolationForest
        pred_if = self.model_iforest.predict(donnees)[0]
        score_iforest = 1.0 if pred_if == -1 else 0.0

        # Buffer temporel
        self.buffer.append([voltage, courant, temperature, soc])
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        score_lstm = 0.0
        if len(self.buffer) == self.buffer_size:
            seq = np.array(self.buffer)
            seq_scaled = self.scaler.transform(seq).reshape(1, self.buffer_size, seq.shape[1])
            score_lstm = float(self.model_lstm.predict(seq_scaled, verbose=0)[0][0])

        score_fusion = float((score_iforest * 0.6) + (score_lstm * 0.4))

        if score_iforest == 1.0:
            decision = "🚨 ANOMALIE DÉTECTÉE - ACTION IMMÉDIATE"
            priorite = "P0_CRITIQUE"
        elif score_lstm > 0.8:
            decision = "⚠️ DÉGRADATION PRÉDITE - PRÉVENTION NÉCESSAIRE"
            priorite = "P1_URGENT"
        elif score_fusion > self.seuil_fusion:
            decision = "⚠️ ATTENTION - SURVEILLANCE RENFORCÉE"
            priorite = "P2_IMPORTANT"
        else:
            decision = "✅ SYSTÈME NOMINAL"
            priorite = "P3_NORMAL"

        return {
            'score_iforest': float(score_iforest),
            'score_lstm': float(score_lstm),
            'score_fusion': float(score_fusion),
            'decision': decision,
            'priorite': priorite
        }

print("\n🔗 Initialisation du système hybride...")
systeme = SystemeHybride(model_iforest, model_lstm, scaler)
print("✅ Système prêt")

# -----------------------------------------------------------------------------#
# PARTIE 5 : TESTS SUR SCÉNARIOS RÉELS
# -----------------------------------------------------------------------------#
scenarios = {
    'nominal': {'description': 'Opération nominale', 'voltage': 7.4, 'courant': 0.5, 'temperature': 25, 'soc': 80},
    'deep_discharge': {'description': 'Deep discharge', 'voltage': 5.8, 'courant': -0.2, 'temperature': -10, 'soc': 5},
    'thermal_runaway': {'description': 'Thermal runaway', 'voltage': 7.6, 'courant': 2.5, 'temperature': 65, 'soc': 85},
    'degradation': {'description': 'Dégradation progressive', 'voltage': 6.8, 'courant': 0.3, 'temperature': 35, 'soc': 55}
}

print("\n🧪 Tests scénarios réels :")
resultats_tests = []
for nom, s in scenarios.items():
    print(f"\n📋 {s['description']}")
    # Remplir le buffer avec valeurs identiques (simule constance)
    for i in range(FENETRE):
        out = systeme.analyser(s['voltage'], s['courant'], s['temperature'], s['soc'])
    print(f"   Score IForest : {out['score_iforest']:.3f}")
    print(f"   Score LSTM    : {out['score_lstm']:.3f}")
    print(f"   Score Fusion  : {out['score_fusion']:.3f}")
    print(f"   Décision      : {out['decision']}")
    print(f"   Priorité      : {out['priorite']}")
    resultats_tests.append(out)

# -----------------------------------------------------------------------------#
# PARTIE 6 : STATISTIQUES & SAUVEGARDE
# -----------------------------------------------------------------------------#
print("\n📊 Performances:")
print(f"   IForest — Acc: {accuracy_if*100:.2f}% | Prec: {precision_if*100:.2f}% | Rec: {recall_if*100:.2f}%")
print(f"   LSTM    — Acc: {test_results[1]*100:.2f}% | Prec: {test_results[2]*100:.2f}% | Rec: {test_results[3]*100:.2f}%")
print(f"   Latences (ms) — IForest: {latence_if:.3f}, LSTM: {latence_lstm:.3f}")

# Sauvegarde
os.makedirs('modeles', exist_ok=True)
model_lstm.save('modeles/model_lstm_optimise.h5')
joblib.dump(model_iforest, 'modeles/model_isolation_forest.pkl')
joblib.dump(scaler, 'modeles/scaler_batterie.pkl')
print("\n💾 Modèles sauvegardés dans dossier ./modeles/")

# -----------------------------------------------------------------------------#
# PARTIE 7 : VISUALISATIONS SIMPLES
# -----------------------------------------------------------------------------#
print("\n📈 Génération des graphiques (aperçu)...")
fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['voltage'][:1000], label='Voltage', alpha=0.7)
ax1.set_title('Voltage (1000 premières mesures)')
ax1.set_ylabel('Voltage (V)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['temperature'][:1000], label='Température', alpha=0.7)
ax2.set_title('Température')
ax2.set_ylabel('Temp (°C)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
ax3.plot(df['soc'][:1000], label='SOC', alpha=0.7)
ax3.set_title('State of Charge')
ax3.set_ylabel('SOC (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 3, 4)
ax4.plot(history.history['loss'], label='Train Loss')
ax4.plot(history.history['val_loss'], label='Val Loss')
ax4.set_title('Loss LSTM')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
ax5.plot(history.history['accuracy'], label='Train Acc')
ax5.plot(history.history['val_accuracy'], label='Val Acc')
ax5.set_title('Accuracy LSTM')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Accuracy')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
cm_if = confusion_matrix(y_test_if, y_pred_if)
ax6.imshow(cm_if, interpolation='nearest')
ax6.set_title('Confusion Matrix - IForest')
ax6.set_xlabel('Prédit')
ax6.set_ylabel('Réel')
for i in range(cm_if.shape[0]):
    for j in range(cm_if.shape[1]):
        ax6.text(j, i, cm_if[i, j], ha='center', va='center', color='red', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('resultats_systeme_hybride.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Graphique sauvegardé : resultats_systeme_hybride.png")

# -----------------------------------------------------------------------------#
# RÉSUMÉ
# -----------------------------------------------------------------------------#
print("\n🎉 SCRIPT TERMINÉ — SYSTÈME HYBRIDE OPÉRATIONNEL")
print("   • Modèles enregistrés dans ./modeles/")
print("   • Visualisation : resultats_systeme_hybride.png")
print("Bonne chance pour le challenge ! 🚀")
