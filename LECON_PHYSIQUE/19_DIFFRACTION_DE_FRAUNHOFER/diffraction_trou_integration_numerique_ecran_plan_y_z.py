import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import time

def tracer_profil_transversal(v_z, v_x, Coupe_Faisceau, z_souhaite):
    """
    Extrait et trace l'intensité transversale en fonction de x 
    pour la distance z la plus proche de 'z_souhaite' (en mètres).
    """
    # 1. Trouver l'indice de la colonne z la plus proche de la valeur demandée
    indice_z = np.argmin(np.abs(v_z - z_souhaite))
    z_reel = v_z[indice_z]
    
    # 2. Extraire la colonne correspondante dans la matrice d'intensité
    # ATTENTION : notre matrice est indexée sous la forme [index_x, index_z]
    profil_x = Coupe_Faisceau[:, indice_z]
    
    # Normalisation locale du profil (le max du profil vaut 1) pour une meilleure lecture
    profil_x_norm = profil_x / np.max(profil_x)
    
    # 3. Tracé graphique du profil
    plt.figure(figsize=(8, 5))
    
    # On convertit v_x en mm pour que l'axe soit plus lisible
    plt.plot(v_x * 1e3, profil_x_norm, color='darkorange', lw=2, label=f"Profil numérique")
    
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.title(f"Coupe transversale du faisceau à $z$ = {z_reel*100:.2f} cm")
    plt.xlabel("Position transversale $x$ (mm)")
    plt.ylabel("Intensité normalisée")
    plt.xlim(v_x.min() * 1e3, v_x.max() * 1e3)
    plt.ylim(-0.05, 1.05)
    
    # Petite astuce pédagogique pour l'oral : si on est loin, on peut comparer à Airy
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return v_x, profil_x_norm

def simuler_faisceau_propre(a, z_max, k, lam, N_z=300, N_x=200, angle_limite_deg=60):
    """
    Version optimisée pour éliminer les artefacts près de la pupille
    en densifiant le maillage radial (N_rho) et en évitant la zone singulière z < 1cm.
    """
    # On commence à 1 cm (0.01 m) : la zone de Fresnel y est
    # et on évite le repliement numérique du plan z=0
    v_z = np.linspace(1e-9, z_max, N_z) 
    taille_x_max = 3.5 * (lam * z_max / a)
    taille_x_max = 0.2*1e-3
    print(f"taille_x_max : {taille_x_max}")
    #taille_x_max = 0.10
    v_x = np.linspace(-taille_x_max/2, taille_x_max/2, N_x)
    
    carte_intensite = np.zeros((N_x, N_z))
    angles_theta = np.zeros((N_x, N_z))
    
    # CRUCIAL : On passe à 300 points sur Phi pour les oscillations de phase proche du trou,
    # et on baisse à 80 sur Theta pour garder un temps de calcul raisonnable (~2-3 secondes).
    N_rho, N_phi = 300, 80
    v_rho = np.linspace(0, a, N_rho)
    v_phi = np.linspace(0, 2 * np.pi, N_phi)
    Rho, Phi = np.meshgrid(v_rho, v_phi)
    X_pup = Rho * np.cos(Phi)
    Y_pup = Rho * np.sin(Phi)

    # Conversion de l'angle limite en radians
    theta_lim = np.radians(angle_limite_deg)
    
    print(f"Calcul des pixels (x, z) avec filtre angulaire à {angle_limite_deg}°...")
    
    for index_x in range(N_x):
        x_M = v_x[index_x]
        for index_z in range(N_z):
            z_M = v_z[index_z]
            
            # --- CALCUL DU MASQUE (LIMITATION ARTEFACT NUMERIQUE LIE AUX GRANDS ANGLES ET VARIATIONS DE PHASE RAPIDES (REPLIEMENT DE SPECTRE)) ---
            # Angle maximal sous lequel le point M voit le bord le plus éloigné du trou
            angle_max_point = np.arctan((np.abs(x_M) + a) / z_M)
            angles_theta[index_x, index_z] = np.degrees(angle_max_point)

            if angle_max_point > theta_lim:
                # Si l'angle est trop grand, la phase oscille trop vite :
                # on n'effectue pas le calcul et on force l'intensité à 0.
                carte_intensite[index_x, index_z] = 0
            else:
                # Sinon, on calcule la physique réelle de Huygens-Fresnel
                R = np.sqrt((x_M - X_pup)**2 + (0 - Y_pup)**2 + z_M**2)

                # Ajout du facteur d'obliquité de Kirchhoff (1 + cos(theta))/2
                cos_theta = z_M / R
                facteur_obliquite = (1 + cos_theta) / 2
                
                integrande = (np.exp(-1j * k * R) / (R * lam)) * Rho * facteur_obliquite
                
                # Intégration
                int_theta = trapezoid(integrande, v_phi, axis=0)
                amplitude_M = trapezoid(int_theta, v_rho)
                
                carte_intensite[index_x, index_z] = np.abs(amplitude_M)**2
            
    return v_z, v_x, carte_intensite, angles_theta

# =============================================================================
# EXÉCUTION
# =============================================================================
if __name__ == '__main__':
    lam = 632.0e-9       
    k = 2 * np.pi / lam  
    a = 10*lam  #rayon du trou        
    
    z_critique = (a**2) / lam
    z_max = 2e-3

    print(f"z critique : {z_critique:.3e}")
    print(f"z max : {z_max:.3e}")
    
    t0 = time.time()
    v_z, v_x, Coupe_Faisceau, angles_theta = simuler_faisceau_propre(a, z_max, k, lam)
    print(f"Calcul réussi en : {time.time() - t0:.2f} secondes.")

    # Normalisation
    Coupe_Faisceau /= np.max(Coupe_Faisceau)

    # 2. On ajoute un petit seuil pour éviter le log(0) et on calcule le log10
    # Une valeur de 1e-4 signifie qu'on verra les détails jusqu'à 4 ordres de grandeur sous le pic max
    seuil_bruit = 1e-4
    Coupe_Log = np.log10(np.clip(Coupe_Faisceau, seuil_bruit, 1.0))

    # Affichage
    plt.figure(figsize=(13, 5))
    extent = [0, z_max * 1e3, -v_x.max() * 1e3, v_x.max() * 1e3]
    
    # Le fait de saturer légèrement à vmax=0.7 permet de voir les structures fines 
    # de Fresnel à gauche sans que les artefacts n'apparaissent.
    plt.imshow(Coupe_Log, extent=extent, cmap='Greys_r', origin='lower', 
               aspect='auto', vmax=0.25)

    # Personnalisation de la barre de couleur pour afficher des puissances de 10
    cbar = plt.colorbar(label="Intensité (Échelle Logarithmique)")
    cbar.set_ticks([-4, -3, -2, -1, 0])
    cbar.set_ticklabels(["10⁻⁴ (0.01%)", "10⁻³ (0.1%)", "10⁻² (1%)", "10⁻¹ (10%)", "1 (100%)"])
    
    plt.axvline(z_critique * 1e3, color='r', linestyle='--', lw=2, 
                label=f"Frontière $z_c$ ({z_critique*1e3:.3e} mm)")
    plt.plot([1*1e-9, z_max*1e3], [a*1000, a*1000], color='b', linestyle='--', alpha=0.7, linewidth=2)
    plt.plot([1*1e-9, z_max*1e3], [-a*1000, -a*1000], color='b', linestyle='--', alpha=0.7, linewidth=2, label="Bords géométriques")
    
    #plt.colorbar(label="Intensité normalisée")
    plt.title(f"Faisceau diffracté, rayon pupille : {a*1e3:.3e} mm, lambda : {lam:.3e} nm")
    plt.xlabel("Axe de propagation z (mm)")
    plt.ylabel("Axe transversal x (mm)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Affichage
    plt.figure(figsize=(13, 5))
    extent = [0, z_max * 1e3, -v_x.max() * 1e3, v_x.max() * 1e3]
    
    # Le fait de saturer légèrement à vmax=0.7 permet de voir les structures fines 
    # de Fresnel à gauche sans que les artefacts n'apparaissent.
    plt.imshow(angles_theta, extent=extent, cmap='Reds', origin='lower', 
               aspect='auto')
    cbar = plt.colorbar(label="Angle")

    plt.show()