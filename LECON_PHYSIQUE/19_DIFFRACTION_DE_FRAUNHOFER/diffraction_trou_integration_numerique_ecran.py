
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def calculer_distance_R(Rho, Theta, x_M, y_M, z):
    """
    Calcule la distance géométrique R entre tous les points de la pupille (coordonnées polaires)
    et un point spécifique M de l'écran d'observation (coordonnées cartésiennes x_M, y_M).
    
    Paramètres :
    ------------
    Rho, Theta : matrices numpy 2D
        Grille de coordonnées polaires de la pupille diffractante.
    x_M, y_M : réels
        Coordonnées cartésiennes du point d'observation M sur l'écran.
    z : réel
        Distance entre la pupille et l'écran d'observation.
        
    Retour :
    --------
    R : matrice numpy 2D
        Distances R pour chaque élément de la pupille.
    """
    # Passage des coordonnées de la pupille en cartésien pour faciliter le calcul de distance 3D
    x_pupille = Rho * np.cos(Theta)
    y_pupille = Rho * np.sin(Theta)
    
    # Formule de la distance euclidienne entre (x_pupille, y_pupille, 0) et (x_M, y_M, z)
    R = np.sqrt((x_M - x_pupille)**2 + (y_M - y_pupille)**2 + z**2)
    return R

def intensite_point_M(x_M, y_M, z, a, k, lam, N_rho=100, N_theta=100, tau0 = 0.010, z0 = 1):
    """
    Calcule l'intensité lumineuse au point M(x_M, y_M) de l'écran par intégration numérique
    directe du principe de Huygens-Fresnel.
    
    Paramètres :
    ------------
    x_M, y_M : réels
        Coordonnées du point d'observation sur l'écran (en mètres).
    z : réel
        Distance de propagation (en mètres).
    a : réel
        Rayon de l'ouverture circulaire (en mètres).
    k : réel
        Norme du vecteur d'onde (2 * pi / lambda).
    N_rho, N_theta : entiers (optionnel)
        Nombre de points de discrétisation pour le maillage de la pupille.
        
    Retour :
    --------
    I : réel
        Intensité lumineuse relative au point M.
    """
    
    # Valable rigoureusement en régime de Fraunhofer
    tau_adapte = tau0 #* (z / z0)**2

    # 1. Discrétisation de l'espace de la pupille (coordonnées polaires)
    v_rho = np.linspace(0, a, N_rho)
    v_theta = np.linspace(0, 2 * np.pi, N_theta)
    Rho, Theta = np.meshgrid(v_rho, v_theta)
    
    # 2. Calcul des distances R entre la pupille et le point M
    R = calculer_distance_R(Rho, Theta, x_M, y_M, z)
    
    cos_theta = z_M / R
    facteur_obliquite = (1 + cos_theta) / 2
    
    # 3. Calcul de l'intégrande (Huygens-Fresnel)
    # L'élément de surface en polaire apporte le facteur multiplicatif 'Rho'
    integrande = facteur_obliquite * (np.exp(-1j * k * R) / (R*lam)) * Rho
    
    # 4. Intégration numérique par la méthode des trapèzes
    pas_rho = a / (N_rho - 1)
    pas_theta = (2 * np.pi) / (N_theta - 1)
    
    # Double intégration : d'abord sur l'axe des lignes (theta), puis sur les colonnes (rho)
    amplitude_complexe = np.trapz(np.trapz(integrande, dx=pas_theta, axis=0), dx=pas_rho)
    
    # 1. L'intensité est le carré du module de l'amplitude complexe multiplie par 0.5
    eclairement_moyen = 0.5*np.abs(amplitude_complexe)**2

    # 2. Énergie totale reçue par pixel (Joules/m²) = Intensité * Temps d'intégration
    energie_recue = eclairement_moyen * tau_adapte

    return energie_recue

# def simuler_diffraction_optim(a, z, k, taille_ecran, N_ecran=200, N_rho=100, N_theta=100):
# 	"""
# 	Calcule la matrice d'intensité sur l'écran de manière ultra-rapide
# 	grâce à une vectorisation complète en 4D avec NumPy.

# 	Paramètres :
# 	------------
# 	a : rayon de la pupille (m)
# 	z : distance de propagation (m)
# 	k : norme du vecteur d'onde (rad/m)
# 	taille_ecran : largeur de la zone d'affichage (m)
# 	N_ecran : résolution de l'écran (pixels de côté)
# 	N_rho, N_theta : résolution du maillage de la pupille
# 	"""
# 	# 1. Discrétisation de la pupille (2D : Rho, Theta) -> Forme (N_theta, N_rho)
# 	v_rho = np.linspace(0, a, N_rho)
# 	v_theta = np.linspace(0, 2 * np.pi, N_theta)
# 	Rho, Theta = np.meshgrid(v_rho, v_theta)

# 	X_pup = Rho * np.cos(Theta)
# 	Y_pup = Rho * np.sin(Theta)

# 	# 2. Discrétisation de l'écran (2D : X_M, Y_M) -> Forme (N_ecran, N_ecran)
# 	v_ecran = np.linspace(-taille_ecran/2, taille_ecran/2, N_ecran)
# 	X_M, Y_M = np.meshgrid(v_ecran, v_ecran)

# 	# 3. TRUC DE VECTORISATION : Extension des matrices en 4D pour calculer toutes les paires simultanément
# 	# On ajoute des dimensions feintes (None) pour forcer le "broadcasting" de NumPy
# 	# Dimensions finales visées : (N_ecran, N_ecran, N_theta, N_rho)
# 	X_pup_4D = X_pup[None, None, :, :]
# 	Y_pup_4D = Y_pup[None, None, :, :]
# 	Rho_4D   = Rho[None, None, :, :]

# 	X_M_4D   = X_M[:, :, None, None]
# 	Y_M_4D   = Y_M[:, :, None, None]

# 	# 4. Calcul instantané de la matrice des distances R en 4D
# 	R = np.sqrt((X_M_4D - X_pup_4D)**2 + (Y_M_4D - Y_pup_4D)**2 + z**2)

# 	# 5. Calcul de l'intégrande sur toute la structure 4D
# 	integrande = (np.exp(-1j * k * R) / R) * Rho_4D

# 	# 6. Double intégration numérique (Trapèzes) sur les axes de la pupille (axes 2 et 3)
# 	pas_rho = a / (N_rho - 1)
# 	pas_theta = (2 * np.pi) / (N_theta - 1)

# 	# On intègre d'abord sur l'axe 2 (theta) puis sur l'axe 3 (rho)
# 	amplitude_complexe = np.trapz(np.trapz(integrande, dx=pas_theta, axis=2), dx=pas_rho, axis=2)

# 	# L'intensité correspond au module au carré
# 	intensite = np.abs(amplitude_complexe)**2

# 	# Normalisation
# 	intensite /= np.max(intensite)
# 	return v_ecran, intensite

# =============================================================================
# SCRIPT PRINCIPAL : PARAMÉTRAGE ET SIMULATION
# =============================================================================
if __name__ == '__main__':
    
    optim = False
    # --- 1. Paramètres physiques (Grandeurs typiques d'un TP d'optique) ---
    lam = 632.8e-9       # Longueur d'onde du laser Hélium-Néon (m)
    k = 2 * np.pi / lam  # Norme du vecteur d'onde (rad/m)
    a = 10 * lam           # Rayon du trou circulaire en m (200 micromètres)
    z = 100*lam             # Distance écran-pupille

    # Distance de Fraunhofer critique : z_f = a^2 / lambda
    z_critique = (a**2) / lam
    
    # CAS A : Régime de Fresnel (Champ proche) -> On choisit z petit devant z_critique
    z_fresnel = 0.5 * z_critique   # 8 cm de la pupille
    
    # CAS B : Régime de Fraunhofer (Champ lointain) -> On choisit z grand devant z_critique
    z_fraunhofer = 2 * z_critique # 1.5 mètre de la pupille
    
    z = 0.2 * z_critique

    print(f"Distance limite théorique de Fraunhofer : {z_critique:.3e} mètres")
    print(f"Rayon du trou: {a:.3e} mètres")
    print(f"Distance écran-pupille {z:.3e} mètres")

    # Définition de l'écran (ajusté selon la distance pour bien voir)
    N_ecran = 90 # Résolution de l'image sur l'écran

    taille_caracteristique = np.sqrt(lam * z) + (lam * z / a)
    
    # On choisit d'observer une zone égale à 4 fois cette taille caractéristique
    taille_ecran = 4 * taille_caracteristique

    #taille_ecran = 2e-3 if z == z_fresnel else 1e-3 # Largeur de la zone observée

    if optim :
    	v_ecran, Image_Intensite = simuler_diffraction_optim(
        a, z, k, taille_ecran, N_ecran=200, N_rho=100, N_theta=100)
    else : 
	    v_ecran = np.linspace(-taille_ecran/2, taille_ecran/2, N_ecran)
	    X_M, Y_M = np.meshgrid(v_ecran, v_ecran)   # Grille 2D représentant l'écran
	    
	    # Matrice qui va stocker l'intensité de chaque pixel de l'écran
	    Image_Intensite = np.zeros((N_ecran, N_ecran))
	    
	    #print("Calcul de la figure de diffraction en cours (Huygens-Fresnel direct)...")
	    # Boucle sur chaque pixel de l'écran
	    for i in range(N_ecran):
	        for j in range(N_ecran):
	            Image_Intensite[i, j] = intensite_point_M(X_M[i, j], Y_M[i, j], z, a, k, lam, tau0=2)
            
    # Normalisation de l'intensité pour avoir un maximum égal à 1 (plus propre pour l'affichage)
    Image_Intensite /= np.max(Image_Intensite)
    #print("Calcul terminé !")

    seuil_bruit = 1e-4
    Image_Intensite_Log = np.log10(np.clip(Image_Intensite, seuil_bruit, 1.0))

    # --- 3. Affichages graphiques avec Matplotlib ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    
    # Graphique 1 : Visualisation 2D de la Tache d'Airy (ce qu'on voit sur l'écran)
    plt.subplot(1, 2, 1)
    # L'extension permet d'afficher les vraies graduations en millimètres (mm)
    extent = [-taille_ecran*1e3/2, taille_ecran*1e3/2, -taille_ecran*1e3/2, taille_ecran*1e3/2]
    
    # Utilisation d'une carte de couleur 'inferno' ou 'gray' pour simuler la lumière
    im = ax[0].imshow(Image_Intensite_Log, extent=extent, cmap='Reds', origin='lower')
    fig.colorbar(im, ax=ax[0], label="Eclairement reçue")
    cercle_pupille = plt.Circle((0, 0), a * 1e3, edgecolor='k', facecolor='none', linestyle='--', linewidth=2, label="Contour de la pupille")
    ax[0].add_patch(cercle_pupille)
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"Figure de diffraction à z = {z:.3e} m, z_fraunhofer : {z_critique:.3e}")
    ax[0].set_xlabel("x (mm)")
    ax[0].set_ylabel("y (mm)")
    
    # Graphique 2 : Coupe transversale (Profil d'intensité selon l'axe horizontal médian)
    ligne_centrale = N_ecran // 2
    ax[1].plot(v_ecran * 1e3, Image_Intensite[ligne_centrale, :], color='k', lw=2, label="Simulation numérique")
    
    ax[1].set_title("Profil d'intensité au centre de l'écran")
    ax[1].set_xlabel("Position x (mm)")
    ax[1].set_ylabel("Eclairement reçue")
    ax[1].grid(True, linestyle='--', alpha=0.7)
    #ax[1].set_xlim(-taille_ecran*1e3/2, taille_ecran*1e3/2)
    #plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()