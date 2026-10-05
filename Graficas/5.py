import matplotlib.pyplot as plt
import numpy as np

# Datos
hilos = [1, 2, 4, 8, 16, 24]
x_pos = np.arange(len(hilos))
bar_width = 0.55

fwd = np.array([48.8850, 24.9947, 12.5973, 6.5515, 3.8310, 2.7356])
bwd = np.array([119.9908, 70.6402, 33.9012, 16.7489, 9.6362, 6.9495])
sgd = np.array([0.2157, 1.3319, 1.6073, 2.1305, 3.5174, 4.9084])
tiempo_total = np.array([169.6415, 99.1022, 52.5712, 27.9934, 19.4477, 17.6482])

tiempo_perfilado = fwd + bwd + sgd
overhead_serie = tiempo_total - tiempo_perfilado

pct_fwd = (fwd / tiempo_total) * 100
pct_bwd = (bwd / tiempo_total) * 100
pct_sgd = (sgd / tiempo_total) * 100
pct_overhead = (overhead_serie / tiempo_total) * 100

# Configuración del gráfico 2
plt.figure(figsize=(9.5, 6))

plt.bar(
    x_pos,
    pct_fwd,
    label="Forward Pass",
    color="#3498db",
    width=bar_width,
    edgecolor="black",
    alpha=0.85,
)
plt.bar(
    x_pos,
    pct_bwd,
    bottom=pct_fwd,
    label="Backward Pass",
    color="#e74c3c",
    width=bar_width,
    edgecolor="black",
    alpha=0.85,
)
plt.bar(
    x_pos,
    pct_sgd,
    bottom=pct_fwd + pct_bwd,
    label="SGD (Sincronización)",
    color="#f39c12",
    width=bar_width,
    edgecolor="black",
    alpha=0.85,
)
plt.bar(
    x_pos,
    pct_overhead,
    bottom=pct_fwd + pct_bwd + pct_sgd,
    label="Overhead Serie / I/O",
    color="#95a5a6",
    width=bar_width,
    edgecolor="black",
    alpha=0.85,
)

plt.title(
    "Distribución Relativa del Tiempo de Ejecución (%)",
    fontsize=14,
    fontweight="bold",
    pad=35,
)
plt.xlabel("Número de Hilos (Threads)", fontsize=12, fontweight="bold")
plt.ylabel("Porcentaje del Tiempo Total (%)", fontsize=12, fontweight="bold")
plt.xticks(x_pos, [str(h) for h in hilos], fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, 118)
plt.grid(True, linestyle="--", alpha=0.4, axis="y")

# Leyenda horizontal posicionada en el margen superior fuera del eje
plt.legend(
    fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False
)

# Etiquetas numéricas arriba de cada columna
for i, p_sgd in enumerate(pct_sgd):
    plt.text(
        i,
        102,
        f"SGD:\n{p_sgd:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#d35400",
    )

plt.tight_layout()
plt.savefig("grafico2_distribucion_corregido.png", dpi=600)
