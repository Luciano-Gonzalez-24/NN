from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 12,
    'axes.titlesize': 13.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUT_DIR = Path(__file__).resolve().parent
C_CUST = '#0072B2'   # Azul: Kernels Custom
C_CUBL = '#009E73'   # Verde: Modular_CUBLAS
C_WARN = '#D55E00'   # Naranja: Setup / Inicialización

# ------------------------------------------------------------------------------
# 1. Fases del entrenamiento según NVTX
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.3))

fases = ['Backward', 'Forward', 'SGD Update']
cust_nvtx = [6.24, 2.23, 0.70]
cubl_nvtx = [8.73, 3.84, 0.71]

x = np.arange(len(fases))
w = 0.35

b1 = ax.bar(x - w/2, cust_nvtx, w, color=C_CUST, label='Kernels Custom (sin cuBLAS)')
b2 = ax.bar(x + w/2, cubl_nvtx, w, color=C_CUBL, label='Modular_CUBLAS (con traza cuBLAS)')

ax.set_title('Tiempo por Fase del Entrenamiento (NVTX)', pad=12)
ax.set_ylabel('Tiempo acumulado [s]')
ax.set_xticks(x)
ax.set_xticklabels(fases, fontweight='bold')
ax.set_ylim(0, 10.2)
ax.legend(frameon=False, loc='upper right')

for bar, col in [(b, C_CUST) for b in b1] + [(b, C_CUBL) for b in b2]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.18, f"{h:.2f} s",
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=col)

plt.tight_layout()
plt.savefig(OUT_DIR / 'prof1_nvtx_fases.pdf')
plt.savefig(OUT_DIR / 'prof1_nvtx_fases.png')
plt.close(fig)

# ------------------------------------------------------------------------------
# 2. Tiempo en GPU por Kernel — Kernels Custom (gpukernsum)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 4.3))

k_names = ['forward_layer', 'grad_weights', 'grad_bias', 'backward_delta', 'sgd_update', 'output_delta']
k_times = [2.69, 2.53, 1.72, 1.00, 0.24, 0.13]
tot_k = sum(k_times)

y = np.arange(len(k_names))
bars = ax.barh(y, k_times, color=C_CUST, height=0.58)
ax.invert_yaxis()
ax.set_yticks(y)
ax.set_yticklabels(k_names, fontweight='bold')
ax.set_xlabel('Tiempo acumulado en GPU [s]')
ax.set_title('Kernels Custom: Tiempo en GPU (Total: 8.30 s)', pad=12)
ax.set_xlim(0, 3.45)

for bar, t in zip(bars, k_times):
    pct = (t / tot_k) * 100
    ax.text(t + 0.06, bar.get_y() + bar.get_height()/2, f"{t:.2f} s ({pct:.0f}%)",
            va='center', fontsize=10.5, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'prof2_gpu_custom.pdf')
plt.savefig(OUT_DIR / 'prof2_gpu_custom.png')
plt.close(fig)

# ------------------------------------------------------------------------------
# 3. Tiempo en GPU por Operación — Modular_CUBLAS (gpukernsum)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 4.1))

cb_names = ['cuBLAS SGEMM', 'cuBLAS SGEMV / Red.', 'Activación + Derivada', 'cuBLAS SAXPY (SGD)', 'output_delta (Custom)']
cb_times = [3.82, 1.49, 1.13, 0.25, 0.13]
tot_cb = sum(cb_times)

y = np.arange(len(cb_names))
bars = ax.barh(y, cb_times, color=C_CUBL, height=0.56)
ax.invert_yaxis()
ax.set_yticks(y)
ax.set_yticklabels(cb_names, fontweight='bold')
ax.set_xlabel('Tiempo acumulado en GPU [s]')
ax.set_title('Modular_CUBLAS: Tiempo en GPU (Total: 6.82 s)', pad=12)
ax.set_xlim(0, 4.85)

for bar, t in zip(bars, cb_times):
    pct = (t / tot_cb) * 100
    ax.text(t + 0.08, bar.get_y() + bar.get_height()/2, f"{t:.2f} s ({pct:.0f}%)",
            va='center', fontsize=10.5, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'prof3_gpu_cublas.pdf')
plt.savefig(OUT_DIR / 'prof3_gpu_cublas.png')
plt.close(fig)

# ------------------------------------------------------------------------------
# 4. Inicialización vs. Entrenamiento Puro (End-to-End sin profiler)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.3))

impls = ['Kernels Custom\n(sin cuBLAS)', 'Modular_CUBLAS\n(con cuBLAS)']
train_t = [9.24, 6.25]
setup_t = [0.20, 4.08]

x = np.arange(len(impls))
w = 0.36

b_tr = ax.bar(x - w/2, train_t, w, color='#2B9348', label='Entrenamiento Puro en GPU')
b_su = ax.bar(x + w/2, setup_t, w, color=C_WARN, label='Setup e Inicialización (Contexto / cuBLAS)')

ax.set_title('Inicialización vs. Entrenamiento Puro (End-to-End)', pad=12)
ax.set_ylabel('Tiempo [s]')
ax.set_xticks(x)
ax.set_xticklabels(impls, fontweight='bold')
ax.set_ylim(0, 10.8)
ax.legend(frameon=False, loc='upper right')

for bar, col in [(b, '#2B9348') for b in b_tr] + [(b, C_WARN) for b in b_su]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.18, f"{h:.2f} s",
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=col)

plt.tight_layout()
plt.savefig(OUT_DIR / 'prof4_setup_vs_train.pdf')
plt.savefig(OUT_DIR / 'prof4_setup_vs_train.png')
plt.close(fig)

print(f"4 gráficos generados en: {OUT_DIR}")