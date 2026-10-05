import matplotlib.pyplot as plt

# Datos de perfilado de tiempos
fases = [
    "Forward Part 1\n(Capas Ocultas)",
    "Forward Part 2\n(Capa Salida)",
    "Backward Part 1\n(Loss & Output)",
    "Backward Part 2\n(Propag. & Pesos)",
]
tiempos = [114.414664, 0.877728, 0.279883, 129.950883]
porcentajes = [46.60, 0.36, 0.11, 52.93]

colores = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

# Configuración de la figura
fig, ax = plt.subplots(figsize=(10, 5))

barras = ax.barh(
    fases, tiempos, color=colores, edgecolor="black", alpha=0.85, height=0.55
)

# Títulos y etiquetas
ax.set_title(
    "Perfilado de Tiempos de Ejecución por Fase (s)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Tiempo (segundos)", fontsize=11, fontweight="bold")
ax.invert_yaxis()  # Mantener secuencia original
ax.grid(True, linestyle="--", alpha=0.4, axis="x")

# Etiquetas con segundos y porcentajes
for barra, pct, t in zip(barras, porcentajes, tiempos):
    ancho = barra.get_width()
    ax.text(
        ancho + 2,
        barra.get_y() + barra.get_height() / 2,
        f"{t:.2f} s  ({pct:.2f}%)",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

ax.set_xlim(0, max(tiempos) * 1.25)

plt.tight_layout()
plt.savefig("perfilado_tiempos.png", dpi=600)
