import matplotlib.pyplot as plt
import numpy as np

# Datos de profiling
hilos = np.array([1, 2, 4, 8, 16, 24])
tiempo_total = np.array([169.6415, 99.1022, 52.5712, 27.9934, 19.4477, 17.6482])
speedup = tiempo_total[0] / tiempo_total

# Cálculo de ganancia marginal (\Delta S / \Delta p)
hilos_mid = []
ganancia = []

for i in range(len(hilos) - 1):
    d_h = hilos[i + 1] - hilos[i]
    d_s = speedup[i + 1] - speedup[i]
    hilos_mid.append(f"{hilos[i]} → {hilos[i+1]}")
    ganancia.append(d_s / d_h)

x_pos = np.arange(len(hilos_mid))

# Configuración del gráfico
plt.figure(figsize=(9, 5.5))
plt.bar(x_pos, ganancia, color="#16a085", edgecolor="black", alpha=0.85, width=0.5)

plt.title(
    "Ganancia Marginal de Speedup por Hilo Adicional",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Salto de Hilos (de $p_1$ a $p_2$)", fontsize=12, fontweight="bold")
plt.ylabel(
    "Speedup Ganado por Hilo Adicional ($ Delta S / Delta p$)",
    fontsize=11,
    fontweight="bold",
)
plt.xticks(x_pos, hilos_mid, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4, axis="y")

# Anotación sobre cada columna
for i, g in enumerate(ganancia):
    plt.text(
        i,
        g + 0.02,
        f"+{g:.2f}x / hilo",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
        color="#0e6251",
    )

plt.ylim(0, max(ganancia) * 1.2)
plt.tight_layout()
plt.savefig("grafico_ganancia_marginal.png", dpi=600)
