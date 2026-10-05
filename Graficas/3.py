import matplotlib.pyplot as plt
import numpy as np

# Datos extraídos
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

speedup_fwd = fwd[0] / fwd
speedup_bwd = bwd[0] / bwd
speedup_global = tiempo_total[0] / tiempo_total


# -------------------------------------------------------------
# GRAFICO 3: SPEEDUP POR FASE VS GLOBAL
# -------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.plot(
    hilos, hilos, "--", color="#7f8c8d", label="Speedup Ideal (Lineal)", linewidth=2
)
plt.plot(
    hilos,
    speedup_fwd,
    marker="o",
    linewidth=2.5,
    color="#3498db",
    label="Forward Pass ($S_{fwd}$)",
    markersize=8,
)
plt.plot(
    hilos,
    speedup_bwd,
    marker="s",
    linewidth=2.5,
    color="#e74c3c",
    label="Backward Pass ($S_{bwd}$)",
    markersize=8,
)
plt.plot(
    hilos,
    speedup_global,
    marker="^",
    linewidth=2.8,
    color="#2ecc71",
    label="Speedup Global ($S_{tot}$)",
    markersize=8,
)

plt.title(
    "Speedup Individual por Fase vs. Speedup Global",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Número de Hilos (Threads)", fontsize=12, fontweight="bold")
plt.ylabel("Aceleración ($T_1 / T_p$)", fontsize=12, fontweight="bold")
plt.xticks(hilos, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=11)

plt.annotate(
    f"{speedup_fwd[-1]:.1f}x",
    (24, speedup_fwd[-1]),
    textcoords="offset points",
    xytext=(-20, 8),
    fontweight="bold",
    fontsize=10,
    color="#2980b9",
)
plt.annotate(
    f"{speedup_bwd[-1]:.1f}x",
    (24, speedup_bwd[-1]),
    textcoords="offset points",
    xytext=(8, -12),
    fontweight="bold",
    fontsize=10,
    color="#c0392b",
)
plt.annotate(
    f"{speedup_global[-1]:.1f}x",
    (24, speedup_global[-1]),
    textcoords="offset points",
    xytext=(8, 5),
    fontweight="bold",
    fontsize=10,
    color="#27ae60",
)

plt.tight_layout()
plt.savefig("grafico3_speedup_fases.png", dpi=600)
