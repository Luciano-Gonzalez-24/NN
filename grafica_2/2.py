from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.35,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Ejecución actual: Epochs = 1, He Uniforme, ReLU, Tiempo = 10.622342 s
preds_1ep = np.array([
    0.939129, 0.981623, 1.004644, 1.014191, 1.017733, 1.015143, 1.012835, 1.012192, 1.013525, 1.012093,
    1.010401, 1.011283, 1.019495, 1.030017, 1.039217, 1.048363, 1.055817, 1.062792, 1.071480, 1.081612,
    1.091838, 1.102230, 1.112622, 1.123014, 1.133406, 1.143798, 1.154189, 1.164582, 1.175017, 1.185520,
    1.196349, 1.208347, 1.220347, 1.232346, 1.244345, 1.256617, 1.268898, 1.281180, 1.295574, 1.313562,
    1.331550, 1.349538, 1.367697, 1.386922, 1.405952, 1.424983, 1.444013, 1.463043, 1.482074, 1.501214,
    1.521273, 1.541860, 1.562453, 1.583723, 1.606281, 1.633136, 1.660409, 1.688703, 1.717012, 1.745324,
    1.773636, 1.801949, 1.830261, 1.858573, 1.886885, 1.915198, 1.943510, 1.971822, 2.000134, 2.028446,
    2.056759, 2.085071, 2.113383, 2.141695, 2.170008, 2.198320, 2.226633, 2.254945, 2.283257, 2.311569,
    2.339882, 2.368193, 2.396506, 2.424818, 2.453130, 2.481442, 2.509755, 2.538067, 2.566380, 2.594692,
    2.623003, 2.651316, 2.679628, 2.707941, 2.736253, 2.764565, 2.792877, 2.821189, 2.849502, 2.877814
])

# Ejecución previa: Epochs = 16, He, ReLU (para comparación)
preds_16ep = np.array([
    0.999494, 1.008404, 1.008003, 1.007454, 1.004636, 1.005723, 1.004617, 1.009109, 1.015841, 1.020149,
    1.024441, 1.028082, 1.032860, 1.037715, 1.040886, 1.048123, 1.057257, 1.066377, 1.075225, 1.083542,
    1.090133, 1.096417, 1.102934, 1.113599, 1.123713, 1.133826, 1.144296, 1.154925, 1.165554, 1.177924,
    1.191374, 1.204824, 1.218273, 1.231723, 1.245173, 1.258622, 1.273611, 1.289462, 1.305645, 1.321868,
    1.338138, 1.354411, 1.370684, 1.386958, 1.403231, 1.419504, 1.435777, 1.453256, 1.475108, 1.496961,
    1.518813, 1.540666, 1.562519, 1.584371, 1.606224, 1.628076, 1.649995, 1.672222, 1.694821, 1.718613,
    1.742532, 1.767893, 1.793254, 1.818615, 1.844127, 1.870365, 1.896603, 1.922864, 1.951144, 1.979425,
    2.007706, 2.035987, 2.064267, 2.092809, 2.121876, 2.153639, 2.185401, 2.218635, 2.252505, 2.286376,
    2.320247, 2.354117, 2.387989, 2.421859, 2.455730, 2.489601, 2.523471, 2.557342, 2.591213, 2.625084,
    2.658955, 2.692825, 2.726696, 2.760567, 2.794438, 2.828308, 2.862180, 2.896050, 2.929921, 2.963792
])

x = np.arange(len(preds_1ep)) * 0.01
y_true = 2.0 * (x ** 2) + 1.0

def metrics(y_pred):
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    r2 = 1.0 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return err, rmse, mae, r2

err1, rmse1, mae1, r2_1 = metrics(preds_1ep)
err16, rmse16, mae16, r2_16 = metrics(preds_16ep)

OUT_DIR = Path(__file__).resolve().parent
x_dense = np.linspace(0, 0.99, 400)
y_dense = 2.0 * (x_dense ** 2) + 1.0

# 1. Gráfica individual: Ajuste con 1 Época (He Uniforme, ReLU)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3), gridspec_kw={'width_ratios': [1.35, 1]})

ax1.plot(x_dense, y_dense, color='#111111', linestyle='--', lw=2.0, label='Función real: $y = 2x^2 + 1$')
ax1.plot(x, preds_1ep, color='#0072B2', lw=2.2, label='Predicción NN (1 época, ReLU)')
ax1.scatter(x[::4], preds_1ep[::4], color='#0072B2', s=22, zorder=4)

box_txt = (
    f"Config: 1 época | He Unif.\n"
    f"Tiempo: 10.62 s (16 hilos)\n"
    f"$R^2 = {r2_1:.5f}$\n"
)
ax1.text(0.04, 0.94, box_txt, transform=ax1.transAxes, fontsize=9.2, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F4F8FB', edgecolor='#0072B2', alpha=0.92))

ax1.set_title('Ajuste de la Red Neuronal (1 Época) vs. Función Objetivo')
ax1.set_xlabel('Entrada ($x$)')
ax1.set_ylabel('Salida ($y$)')
ax1.set_xlim(-0.02, 1.02)
ax1.legend(loc='lower right', frameon=True)

ax2.axhline(0, color='#111111', linestyle='--', lw=1.3)
ax2.fill_between(x, 0, err1, where=(err1 >= 0), interpolate=True, color='#009E73', alpha=0.22)
ax2.fill_between(x, 0, err1, where=(err1 < 0), interpolate=True, color='#D55E00', alpha=0.25)
ax2.plot(x, err1, color='#D55E00', lw=2.0, label='Error residual ($\\hat{y} - y$)')

idx_pos, idx_neg = np.argmax(err1), np.argmin(err1)
ax2.scatter([x[0], x[idx_pos], x[idx_neg]], [err1[0], err1[idx_pos], err1[idx_neg]],
            color='#D55E00', s=42, zorder=5, ec='black', lw=0.6)
ax2.annotate(f"$x=0.00$: {err1[0]:+.4f}", xy=(x[0], err1[0]), xytext=(10, -4),
             textcoords='offset points', ha='left', va='center', fontsize=8.5, fontweight='bold', color='#D55E00')
ax2.annotate(f"+{err1[idx_pos]:.4f}\n($x={x[idx_pos]:.2f}$)", xy=(x[idx_pos], err1[idx_pos]), xytext=(8, 2),
             textcoords='offset points', ha='left', va='bottom', fontsize=8.5, fontweight='bold', color='#007A58')
ax2.annotate(f"Máx: {err1[idx_neg]:+.4f}\n($x={x[idx_neg]:.2f}$)", xy=(x[idx_neg], err1[idx_neg]), xytext=(-12, 2),
             textcoords='offset points', ha='right', va='bottom', fontsize=8.5, fontweight='bold', color='#D55E00')

ax2.set_title('Error de Aproximación ($\\hat{y}_{\\mathrm{NN}} - y_{\\mathrm{real}}$)')
ax2.set_xlabel('Entrada ($x$)')
ax2.set_ylabel('Error residual')
ax2.set_xlim(-0.02, 1.04)
ax2.set_ylim(-0.098, 0.112)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_ajuste_1epoca.pdf')
plt.savefig(OUT_DIR / 'fig_ajuste_1epoca.png')

# 2. Gráfica comparativa: 1 Época vs 16 Épocas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3), gridspec_kw={'width_ratios': [1.3, 1]})

ax1.plot(x_dense, y_dense, color='#111111', linestyle='--', lw=2.0, label='Función real: $y = 2x^2 + 1$')
ax1.plot(x, preds_1ep, color='#E69F00', lw=2.2, label=f"1 Época ($R^2={r2_1:.4f}$, RMSE={rmse1:.3f})")
ax1.plot(x, preds_16ep, color='#0072B2', lw=2.2, label=f"16 Épocas ($R^2={r2_16:.4f}$, RMSE={rmse16:.3f})")
ax1.set_title('Convergencia del Ajuste: 1 Época vs. 16 Épocas')
ax1.set_xlabel('Entrada ($x$)')
ax1.set_ylabel('Salida ($y$)')
ax1.set_xlim(-0.02, 1.02)
ax1.legend(loc='upper left', frameon=True)

ax2.axhline(0, color='#111111', linestyle='--', lw=1.3)
ax2.plot(x, err1, color='#E69F00', lw=2.1, label='Residuo 1 Época')
ax2.plot(x, err16, color='#0072B2', lw=2.1, label='Residuo 16 Épocas')
ax2.set_title('Reducción del Error Residual por Épocas')
ax2.set_xlabel('Entrada ($x$)')
ax2.set_ylabel('Error residual ($\\hat{y} - y$)')
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.098, 0.105)
ax2.legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_comparacion_epocas.pdf')
plt.savefig(OUT_DIR / 'fig_comparacion_epocas.png')
print(f"Gráficas guardadas en: {OUT_DIR}")