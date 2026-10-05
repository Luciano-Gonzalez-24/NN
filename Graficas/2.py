import matplotlib.pyplot as plt
import numpy as np

# Datos recolectados del profiling
hilos = np.array([1, 2, 4, 8, 16, 24])
tiempo_total = np.array(
    [169.641476, 99.102288, 52.571280, 27.993485, 19.447819, 17.648274]
)

# Cálculo de Métricas
speedup_real = tiempo_total[0] / tiempo_total
speedup_ideal = hilos
eficiencia = (speedup_real / hilos) * 100

# Configuración del Canvas (2 subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Gráfico 1: Speedup ---
ax1.plot(
    hilos,
    speedup_real,
    marker="o",
    linewidth=2.5,
    color="#2b5c8f",
    label="Speedup Real",
    markersize=7,
)
ax1.plot(
    hilos,
    speedup_ideal,
    linestyle="--",
    color="#e74c3c",
    label="Speedup Ideal (Lineal)",
    linewidth=2,
)

ax1.set_title("Aceleración (Speedup $S(p)$)", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlabel("Número de Hilos (Threads)", fontsize=11)
ax1.set_ylabel("Speedup ($T_1 / T_p$)", fontsize=11)
ax1.set_xticks(hilos)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# Etiquetas en cada punto de Speedup
for h, s in zip(hilos, speedup_real):
    ax1.annotate(
        f"{s:.2f}x",
        (h, s),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
        fontsize=9.5,
    )

# --- Gráfico 2: Eficiencia ---
ax2.plot(
    hilos,
    eficiencia,
    marker="s",
    linewidth=2.5,
    color="#2ecc71",
    label="Eficiencia Real",
    markersize=7,
)
ax2.axhline(
    100, linestyle="--", color="#e74c3c", label="Eficiencia Ideal (100%)", linewidth=2
)

ax2.set_title("Eficiencia Paralela ($E(p)$)", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("Número de Hilos (Threads)", fontsize=11)
ax2.set_ylabel("Eficiencia (%)", fontsize=11)
ax2.set_xticks(hilos)
ax2.set_ylim(0, 115)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

# Etiquetas en cada punto de Eficiencia
for h, e in zip(hilos, eficiencia):
    ax2.annotate(
        f"{e:.1f}%",
        (h, e),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
        fontsize=9.5,
    )

plt.tight_layout()
plt.savefig("speedup_eficiencia.png", dpi=600)
