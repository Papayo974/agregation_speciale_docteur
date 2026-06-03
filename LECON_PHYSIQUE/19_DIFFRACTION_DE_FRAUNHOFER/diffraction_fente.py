import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def simuler_diffraction_fente_rectangulaire(a, b, lam, z, taille_ecran=0.005, N_points=500):
    """
    Simule la figure de diffraction de Fraunhofer pour une fente de dimensions a x b
    en utilisant les fonctions sinus cardinaux.
    """
    # 1. Création de la grille de l'écran (en mètres)
    v_x = np.linspace(-taille_ecran/2, taille_ecran/2, N_points)
    v_y = np.linspace(-taille_ecran/2, taille_ecran/2, N_points)
    X, Y = np.meshgrid(v_x, v_y)
    
    # 2. Calcul des arguments des sinus cardinaux
    # En Python, np.sinc(x) calcule sin(pi*x) / (pi*x), le facteur pi est déjà inclus !
    arg_x = (a * X) / (lam * z)
    arg_y = (b * Y) / (lam * z)
    
    # 3. Calcul de l'amplitude puis de l'intensité
    amplitude = np.sinc(arg_x) * np.sinc(arg_y)
    intensite = amplitude**2
    
    return v_x, v_y, intensite

# =============================================================================
# PARAMÈTRES PHYSIQUES ET AFFICHAGE
# =============================================================================
if __name__ == '__main__':
    # Paramètres de l'expérience (en mètres)
    lam = 507.e-9       # Laser Hélium-Néon (Rouge)
    z = 1.            # Distance écran-fente (1 mètre)
    
    # Dimensions de la fente (Exemple : fente verticale très fine)
    a = 100e-6           # Largeur selon x = 150 µm (petite dimension -> grande diffraction)
    b = 1e-1          # Hauteur selon y = 600 µm (grande dimension -> petite diffraction)
    
    # Simulation
    v_x, v_y, Intensite = simuler_diffraction_fente_rectangulaire(a, b, lam, z, taille_ecran=0.005)
    
    # Préparation de l'échelle log pour voir les lobes secondaires faibles
    seuil_bruit = 1e-4
    Intensite_Log = np.log10(np.clip(Intensite, seuil_bruit, 1.0))
    
    # --- TRACÉ GRAPHIQUE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    extent = [v_x.min()*1e3, v_x.max()*1e3, v_y.min()*1e3, v_y.max()*1e3] # en mm
    
    # Graphe 1 :
    im1 = ax1.imshow(Intensite, extent=extent, cmap='gray', origin='lower')
    ax1.set_title("Intensité en échelle Linéaire")
    ax1.set_xlabel("Position $x$ sur l'écran (mm)")
    ax1.set_ylabel("Position $y$ sur l'écran (mm)")
    fig.colorbar(im1, ax=ax1, label="Intensité normalisée")

    # AJOUT DU RECTANGLE DE LA FENTE (Converti en mm pour correspondre à l'axe)
    # Le coin inférieur gauche du rectangle doit être centré : (x_centre - largeur/2, y_centre - hauteur/2)
    largeur_fente_mm = a * 1e3
    hauteur_fente_mm = b * 1e3
    
    rect = patches.Rectangle(
        (-largeur_fente_mm / 2, -hauteur_fente_mm / 2),  # Position de départ (x, y)
        largeur_fente_mm,                                # Largeur du rectangle
        hauteur_fente_mm,                                # Hauteur du rectangle
        linewidth=1.5, 
        edgecolor='r',                                # Couleur contrastée pour le jury
        facecolor='none',                                # Ne pas remplir pour voir la lumière derrière
        linestyle='-',
        label=f"Fente ({largeur_fente_mm:.1f}x{hauteur_fente_mm:.1f} mm)"
    )
    
    
    # Graphe 2 : 
    im2 = ax2.imshow(Intensite_Log, extent=extent, cmap='Greys_r', origin='lower')
    ax2.set_title("Intensité en échelle Logarithmique (vue à l'oeil)")
    ax2.set_xlabel("Position $x$ sur l'écran (mm)")
    ax2.set_ylabel("Position $y$ sur l'écran (mm)")
    ax2.add_patch(rect)
    ax2.legend(loc="upper right") # Pour afficher l'étiquette de la fente
    
    cbar = fig.colorbar(im2, ax=ax2, label="Intensité relative")
    cbar.set_ticks([-4, -3, -2, -1, 0])
    cbar.set_ticklabels(["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"])
    
    plt.suptitle(f"Diffraction de Fraunhofer - Fente Rectangulaire ({a*1e6:.0f} µm x {b*1e6:.0f} µm)", fontsize=14)
    plt.tight_layout()
    plt.show()