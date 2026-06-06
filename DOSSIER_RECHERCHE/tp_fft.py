
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 14,  # Taille générale du texte
        "axes.titlesize": 16,  # Taille du titre des graphiques
        "axes.labelsize": 15,  # Taille des labels (X et Y)
        "xtick.labelsize": 12,  # Taille des chiffres sur l'axe X
        "ytick.labelsize": 12,  # Taille des chiffres sur l'axe Y
        "legend.fontsize": 13,  # Taille du texte de la légende
    }
)

# =============================================================================
# 1. PARAMÈTRES DE L'OSCILLOSCOPE ET DU SIGNAL
# =============================================================================

# Paramètres du signal (GBF)
f_signal = 1000  # Fréquence du signal en Hz (ex: 1 kHz)

# Paramètres de l'oscilloscope (Similaires à un oscillo classique)
N = 2500  # Nombre de points fixe de la mémoire
nb_divisions = 10  # L'écran possède 10 divisions horizontales

# PARAMÈTRE À MODIFIER PAR L'ÉLÈVE 
calibre = 200e-3  # Reglage manuel du calibre temporel en s/div (ex: 100 µs/div)

fe_cible = 3 * f_signal

calibre = N / (fe_cible * nb_divisions) # Relation liant le calibre à N, à la frequence d'échantillonnage souhaitee et au nb de divisions.

# =============================================================================
# 2. CALCULS DES PARAMÈTRES D'ACQUISITION
# ========================== ===================================================

T_acq = nb_divisions * calibre  # Durée totale de l'acquisition (s)
f_e = N / T_acq  # Fréquence d'échantillonnage (Hz)
delta_f = 1 / T_acq  # Résolution spectrale (Hz)

print("-" * 50)
print(f"Calibre : {calibre*1e3:.2f} ms/div")
print(f"Durée d'acquisition (Tacq) : {T_acq*1e3:.2f} ms")
print(f"Fréquence d'échantillonnage (fe) : {f_e/1e3:.2f} kHz")
print(f"Fréquence de Nyquist (fe/2) : {f_e/2/1e3:.2f} kHz")
print(f"Résolution spectrale : {delta_f:.2f} Hz")
print("-" * 50)


# =============================================================================
# PARAMETRES D'ACQUISITION POUR LE SIGNAL DE REFERENCE
# ========================== ===================================================
fe_ref = 100 * f_signal
T_acq_ref = 2
N_ref = int(fe_ref * T_acq_ref)
delta_f_ref = 1 / T_acq_ref 


print("-" * 50)
print(f"Durée d'acquisition ref (Tacq ref) : {T_acq_ref*1e3:.2f} ms")
print(f"Fréquence d'échantillonnage ref (fe ref) : {fe_ref/1e3:.2f} kHz")
print(f"Fréquence de Nyquist (fe ref/2) : {fe_ref/2/1e3:.2f} kHz")
print(f"Résolution spectrale ref : {delta_f_ref:.2f} Hz")
print("-" * 50)

# Vérification rapide du théorème de Shannon
if f_e < 2 * f_signal:
    print("⚠ATTENTION : Risque de repliement de spectre (fe < 2*f_signal) !")

# =============================================================================
# 3. GÉNÉRATION DU SIGNAL CONTINU ET ÉCHANTILLONNÉ
# =============================================================================

# Vecteur temps (N points de 0 à T_acq)
t_echan = np.linspace(0, T_acq, N)
t = np.linspace(0, T_acq_ref, N_ref)


y_echan = np.sin(2 * np.pi * f_signal * t_echan)
y = np.sin(2 * np.pi * f_signal * t) 

y_batt = 0.5*(np.sin(2 * np.pi * f_signal * t) + np.sin(2 * np.pi * (f_signal+1.61) * t))
 
#y_echan = np.sin(2 * np.pi * f_signal * t_echan) + np.sin(0.5 * 2 * np.pi * f_signal * t_echan)
#y = np.sin(2 * np.pi * f_signal * t) + np.sin(0.5 * 2 * np.pi * f_signal * t)


# =============================================================================
# 4. CALCUL DE LA FFT (SPECTRE)
# =============================================================================

# Calcul de la FFT et normalisation
spectre_complet = np.fft.fft(y_echan) / N
# On ne garde que la première moitié du spectre (fréquences positives)
spectre_utile = np.abs(spectre_complet[: N // 2]) * 2
phase = np.angle(spectre_complet[: N // 2])

# Vecteur des fréquences associées
frequences = np.fft.fftfreq(N, d=1 / f_e)[: N // 2]

# Calcul de la FFT et normalisation de référence
spectre_complet_ref = np.fft.fft(y) / N_ref
# On ne garde que la première moitié du spectre (fréquences positives)
spectre_utile_ref = np.abs(spectre_complet_ref[: N_ref // 2]) * 2
phase_ref = np.angle(spectre_complet_ref[: N_ref // 2])

# Vecteur des fréquences associées
frequences_ref = np.fft.fftfreq(N_ref, d=1 / fe_ref)[: N_ref // 2]

# =============================================================================
# 5. TRACÉS GRAPHIQUES
# =============================================================================

plt.figure(figsize=(12, 5))

# Graphique 1 : Signal temporel (ce qu'on voit à l'écran)
plt.subplot(1, 2, 1)
plt.plot(t * 1e3, y, "bx-", label="Signal référence", markersize=5)
#plt.plot(t * 1e3, y_batt, "gx-", label="Signal battement", markersize=5)
plt.plot(t_echan * 1e3, y_echan, "ro-", label="Signal échantillonné", markersize=10)

plt.title("Signal temporel (Écran de l'oscilloscope)")
plt.xlabel("Temps (ms)")
plt.ylabel("Tension (V)")
plt.grid(True)
plt.legend()

# Graphique 2 : Spectre en fréquence (FFT)
plt.subplot(1, 2, 2)
# On trace des barres (stem) pour bien voir le côté discret de la FFT
container = plt.stem(frequences / 1e3, spectre_utile/spectre_utile.max(), basefmt=" ", linefmt="r", label="fft echantillonnage")
container.markerline.set_markersize(8)
markerline, stemlines, baseline = plt.stem(frequences_ref / 1e3, spectre_utile_ref/spectre_utile_ref.max(), basefmt=" ", markerfmt='D', linefmt="b", label="fft référence")
markerline.set_markerfacecolor("none")
markerline.set_markersize(8)
plt.title("Analyse spectrale (FFT)")
plt.xlabel("Fréquence (kHz)")
plt.ylabel("Amplitude")
plt.xlim(0, fe_ref/2 /1e3)  # On s'arrête à la fréquence de Nyquist
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()