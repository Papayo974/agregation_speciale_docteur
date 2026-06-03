
import numpy as np
import matplotlib.pyplot as plt

class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


# 1. Génération de données simulées
t_initial = np.linspace(0, 5, 6)
temp_initial = 20.1 + 0.02 * t_initial  # Légère dérive initiale

t_combustion = np.linspace(5, 12, 15)
# Simulation d'une courbe en S pour la combustion
temp_combustion = 20.2 + 8 * (1 - np.exp(-0.5 * (t_combustion - 5))) 

t_final = np.linspace(12, 20, 9)
temp_final = 27.5 - 0.15 * (t_final - 12)  # Refroidissement linéaire

# 2. Calculs pour la correction
t_median = 8.5 # Instant où la température est à mi-hauteur
# Extrapolations (droites de tendance)
pente_init = np.polyfit(t_initial, temp_initial, 1)
pente_fin = np.polyfit(t_final, temp_final, 1)

# Points d'intersection à t_median
y_bas = np.polyval(pente_init, t_median)
y_haut = np.polyval(pente_fin, t_median)
delta_t_corrige = y_haut - y_bas

# 3. Tracé du graphique
fig, ax = plt.subplots()
plt.figure(figsize=(10, 6))
plt.plot(t_initial, temp_initial, 'bo', label='Phase initiale')
ax.plot(t_combustion, temp_combustion, 'go', label='Combustion')
plt.plot(t_final, temp_final, 'ro', label='Phase finale')

# Tracer les droites d'extrapolation
t_extrapol = np.linspace(0, 20, 100)
plt.plot(t_extrapol, np.polyval(pente_init, t_extrapol), 'b--', alpha=0.5)
plt.plot(t_extrapol, np.polyval(pente_fin, t_extrapol), 'r--', alpha=0.5)

# Tracer la correction au temps médian
plt.vlines(t_median, y_bas, y_haut, colors='black', linestyles='solid', lw=2)
plt.annotate(f'ΔT corrigé = {delta_t_corrige:.2f}°C', xy=(t_median, (y_bas + y_haut)/2), 
             xytext=(t_median+1, (y_bas + y_haut)/2), arrowprops=dict(arrowstyle='->'))

plt.title('Correction de Regnault-Pfaundler (Calorimétrie)')
plt.xlabel('Temps (min)')
plt.ylabel('Température (°C)')
plt.legend()
plt.grid(True, linestyle=':')
cursor = Cursor(ax)
fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
plt.show()