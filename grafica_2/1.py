from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# Estilo optimizado para diapositivas Beamer
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

C_MKL = '#D55E00'
C_HIB = '#0072B2'
C_BATCHES = {256: '#CC79A7', 1024: '#E69F00', 8192: '#009E73', 16384: '#0072B2'}
C_PHASES = {'forward': '#0072B2', 'backward': '#56B4E9', 'update': '#D55E00', 'resto': '#999999'}
hilos_ticks = [1, 2, 4, 8, 12, 16, 24]

# Buscar los CSV de benchmarks en la carpeta del script, el CWD o la carpeta padre
SCRIPT_DIR = Path(__file__).resolve().parent
search_dirs = [SCRIPT_DIR, Path.cwd(), SCRIPT_DIR.parent]

csv_files = []
for d in search_dirs:
    found = [
        f for f in sorted(d.glob("*.csv"))
        if re.search(r'batch[_-]?\d+', f.name, re.IGNORECASE)
    ]
    if found:
        csv_files = found
        break

if not csv_files:
    raise FileNotFoundError(
        f"No se encontraron archivos CSV con 'batch<N>' en: {[str(d) for d in search_dirs]}"
    )

def parse_filename(fname):
    m_batch = re.search(r'batch[_-]?(\d+)', fname, re.IGNORECASE)
    if not m_batch:
        return None
    es_hibrido = bool(re.search(r'hib[rt]ido', fname, re.IGNORECASE))
    estrategia = 'Híbrido (Data-Parallel)' if es_hibrido else 'Solo MKL (MatMul)'
    batch = int(m_batch.group(1))
    m_capas = re.search(r'capas[_-]?2\^(\d+)', fname, re.IGNORECASE)
    capas = 2 ** int(m_capas.group(1)) if m_capas else 64
    return estrategia, es_hibrido, batch, capas

records = []
for f in csv_files:
    parsed = parse_filename(f.name)
    if parsed is None:
        print(f"[Omitido] Nombre sin formato esperado: {f.name}")
        continue
    df = pd.read_csv(f)
    if not {'hilos', 'total'}.issubset(df.columns):
        print(f"[Omitido] Columnas incompatibles en: {f.name}")
        continue

    estrategia, es_hibrido, batch, capas = parsed
    t1 = df.loc[df['hilos'] == 1, 'total'].values[0]
    df['estrategia'] = estrategia
    df['es_hibrido'] = es_hibrido
    df['batch'] = batch
    df['capas'] = capas
    df['speedup'] = t1 / df['total']
    df['eficiencia'] = (df['speedup'] / df['hilos']) * 100.0
    df['pct_update'] = (df['update'] / df['total']) * 100.0
    records.append(df)
    print(f"[OK] Cargado: {f.name} -> {estrategia}, Batch={batch}, Capas={capas}x{capas}")

if not records:
    raise ValueError("Ningún CSV válido pudo cargarse para graficar.")

all_df = pd.concat(records, ignore_index=True)

# 1. Comparación directa Solo MKL vs Híbrido
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
pairs = [
    (256,  64,  'Capas 64×64 ($2^6$)\nBatch = 256',    [2, 5, 10, 20, 50, 100, 200]),
    (1024, 128, 'Capas 128×128 ($2^7$)\nBatch = 1024', [5, 10, 20, 50]),
    (1024, 256, 'Capas 256×256 ($2^8$)\nBatch = 1024', [15, 25, 50, 100, 160]),
]

for ax, (b, c, title, yticks) in zip(axes, pairs):
    sub_mkl = all_df[(~all_df['es_hibrido']) & (all_df['batch'] == b) & (all_df['capas'] == c)].sort_values('hilos')
    sub_hib = all_df[(all_df['es_hibrido'])  & (all_df['batch'] == b) & (all_df['capas'] == c)].sort_values('hilos')

    ax.plot(sub_mkl['hilos'], sub_mkl['total'], marker='s', lw=2.2, ms=6, color=C_MKL, label='Solo MKL (MatMul)')
    ax.plot(sub_hib['hilos'], sub_hib['total'], marker='o', lw=2.2, ms=6, color=C_HIB, label='Híbrido (Batch + Grad. local)')

    opt_mkl = sub_mkl.loc[sub_mkl['total'].idxmin()]
    opt_hib = sub_hib.loc[sub_hib['total'].idxmin()]
    ax.scatter([opt_mkl['hilos']], [opt_mkl['total']], s=160, marker='*', color=C_MKL, zorder=5, ec='black', lw=0.6)
    ax.scatter([opt_hib['hilos']], [opt_hib['total']], s=160, marker='*', color=C_HIB, zorder=5, ec='black', lw=0.6)

    ax.annotate(f"Ópt: {opt_mkl['total']:.2f}s\n({int(opt_mkl['hilos'])}h)",
                xy=(opt_mkl['hilos'], opt_mkl['total']), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=8.5, color=C_MKL, fontweight='bold')
    ax.annotate(f"Ópt: {opt_hib['total']:.2f}s\n({int(opt_hib['hilos'])}h)",
                xy=(opt_hib['hilos'], opt_hib['total']), xytext=(0, -23),
                textcoords='offset points', ha='center', fontsize=8.5, color=C_HIB, fontweight='bold')

    t24_mkl = sub_mkl.loc[sub_mkl['hilos'] == 24, 'total'].values[0]
    if t24_mkl > opt_mkl['total'] * 1.15:
        ax.annotate(f"{t24_mkl:.1f}s", xy=(24, t24_mkl), xytext=(-4, 6),
                    textcoords='offset points', ha='right', fontsize=8.5, color=C_MKL)

    ax.set_yscale('log')
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title(title)
    ax.set_xlabel('Número de hilos')
    ax.set_xticks(hilos_ticks)
    ax.margins(y=0.24)

axes[0].set_ylabel('Tiempo total [s] (escala log)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'fig1_mkl_vs_hibrido.pdf')
plt.savefig(SCRIPT_DIR / 'fig1_mkl_vs_hibrido.png')

# 2. Escalabilidad del enfoque Híbrido según Batch (Capas 64x64)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4))
markers_b = {256: '^', 1024: 's', 8192: 'D', 16384: 'o'}
ax1.plot(hilos_ticks, hilos_ticks, linestyle=':', color='black', lw=1.6, label='Speedup ideal ($S = p$)')
ax2.axhline(100, linestyle=':', color='black', lw=1.6, label='Eficiencia ideal (100%)')

for b in [256, 1024, 8192, 16384]:
    sub = all_df[(all_df['es_hibrido']) & (all_df['capas'] == 64) & (all_df['batch'] == b)].sort_values('hilos')
    ax1.plot(sub['hilos'], sub['speedup'], marker=markers_b[b], lw=2.2, ms=6, color=C_BATCHES[b], label=f'Batch = {b}')
    ax2.plot(sub['hilos'], sub['eficiencia'], marker=markers_b[b], lw=2.2, ms=6, color=C_BATCHES[b], label=f'Batch = {b}')
    s24, e24 = sub.loc[sub['hilos'] == 24, ['speedup', 'eficiencia']].values[0]
    dy = -10 if b == 1024 else 4
    ax1.annotate(f"{s24:.1f}×", xy=(24, s24), xytext=(5, dy), textcoords='offset points', fontsize=8.5, fontweight='bold', color=C_BATCHES[b])
    ax2.annotate(f"{e24:.0f}%", xy=(24, e24), xytext=(5, dy), textcoords='offset points', fontsize=8.5, fontweight='bold', color=C_BATCHES[b])

ax1.set_title('Speedup ($T_1 / T_p$) — Capas 64×64 (Híbrido)')
ax1.set_xlabel('Número de hilos ($p$)')
ax1.set_ylabel('Speedup')
ax1.set_xticks(hilos_ticks)
ax1.set_xlim(0.5, 26.8)
ax1.set_ylim(0, 25.5)

ax2.set_title('Eficiencia Paralela ($S_p / p$) — Capas 64×64 (Híbrido)')
ax2.set_xlabel('Número de hilos ($p$)')
ax2.set_ylabel('Eficiencia [%]')
ax2.set_xticks(hilos_ticks)
ax2.set_xlim(0.5, 26.8)
ax2.set_ylim(0, 115)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=True)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'fig2_escalabilidad_batch_hibrido.pdf')
plt.savefig(SCRIPT_DIR / 'fig2_escalabilidad_batch_hibrido.png')

# 3. Desglose por fases y cuello de botella en Update
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
sub_mkl_128 = all_df[(~all_df['es_hibrido']) & (all_df['batch'] == 1024) & (all_df['capas'] == 128)].sort_values('hilos')
x = np.arange(len(sub_mkl_128))
bottom = np.zeros(len(sub_mkl_128))
for phase, lbl in [('forward', 'Forward'), ('backward', 'Backward'), ('update', 'Update'), ('resto', 'Resto')]:
    vals = sub_mkl_128[phase].values
    ax1.bar(x, vals, 0.58, bottom=bottom, color=C_PHASES[phase], label=lbl)
    bottom += vals
for idx, tot in enumerate(bottom):
    ax1.text(idx, tot + 1.0, f"{tot:.1f}s", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_title('Solo MKL: Batch 1024, Capas 128×128\n(Contención de Forward > 8 hilos)')
ax1.set_xticks(x)
ax1.set_xticklabels(sub_mkl_128['hilos'].astype(str))
ax1.set_xlabel('Número de hilos')
ax1.set_ylabel('Tiempo acumulado [s]')
ax1.set_ylim(0, 61)
ax1.legend(loc='upper right', frameon=True)

for b, style in [(256, '-'), (1024, '--'), (8192, '-.'), (16384, '-')]:
    sub = all_df[(all_df['es_hibrido']) & (all_df['capas'] == 64) & (all_df['batch'] == b)].sort_values('hilos')
    ax2.plot(sub['hilos'], sub['pct_update'], marker='o', linestyle=style, lw=2.2, color=C_BATCHES[b], label=f'Batch = {b}')
    u24 = sub.loc[sub['hilos'] == 24, 'pct_update'].values[0]
    ax2.annotate(f"{u24:.1f}%", xy=(24, u24), xytext=(5, -3), textcoords='offset points', fontsize=8.5, fontweight='bold', color=C_BATCHES[b])

ax2.set_title('Híbrido: Peso relativo de Update (Reducción de Gradientes)\n(Cuello de botella secuencial en batches pequeños)')
ax2.set_xlabel('Número de hilos')
ax2.set_ylabel('% del tiempo total en Update')
ax2.set_xticks(hilos_ticks)
ax2.set_xlim(0.5, 26.8)
ax2.set_ylim(0, 46)
ax2.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'fig3_desglose_fases.pdf')
plt.savefig(SCRIPT_DIR / 'fig3_desglose_fases.png')

# 4. Resumen ejecutivo de tiempos óptimos y speedups máximos
summary = []
for (estrategia, batch, capas), g in all_df.groupby(['estrategia', 'batch', 'capas']):
    best = g.loc[g['total'].idxmin()]
    summary.append({
        'estrategia': estrategia,
        'etiqueta': f"{'Híbrido' if 'Híbrido' in estrategia else 'MKL'} | B={batch} | {capas}×{capas}",
        'capas': capas, 'batch': batch,
        'mejor_t': best['total'], 'hilos_opt': int(best['hilos']),
        'speedup_max': best['speedup'], 'efic_opt': best['eficiencia']
    })
sum_plot = pd.DataFrame(summary).sort_values(['capas', 'batch', 'estrategia'], ascending=[False, False, True]).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
colors_bar = [C_HIB if 'Híbrido' in s else C_MKL for s in sum_plot['estrategia']]
y_pos = np.arange(len(sum_plot))

bars1 = ax1.barh(y_pos, sum_plot['mejor_t'], color=colors_bar, height=0.65)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(sum_plot['etiqueta'], fontsize=9.5)
ax1.set_xlabel('Mejor tiempo de ejecución [s]')
ax1.set_title('Tiempo Óptimo Alcanzado')
ax1.set_xlim(0, 31)
for bar, (_, r) in zip(bars1, sum_plot.iterrows()):
    ax1.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2, f"{r['mejor_t']:.2f}s ({r['hilos_opt']}h)", va='center', fontsize=8.5, fontweight='bold')

bars2 = ax2.barh(y_pos, sum_plot['speedup_max'], color=colors_bar, height=0.65)
ax2.set_xlabel('Speedup máximo respecto a 1 hilo')
ax2.set_title('Speedup Máximo Alcanzado')
ax2.set_xlim(0, 25.5)
for bar, (_, r) in zip(bars2, sum_plot.iterrows()):
    ax2.text(bar.get_width() + 0.35, bar.get_y() + bar.get_height()/2, f"{r['speedup_max']:.2f}× ({r['efic_opt']:.0f}% ef.)", va='center', fontsize=8.5, fontweight='bold')

fig.legend(handles=[Patch(facecolor=C_HIB, label='Híbrido (Data-Parallel + Grad. local)'),
                    Patch(facecolor=C_MKL, label='Solo MKL (Paralelismo en MatMul)')],
           loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'fig4_resumen_optimos.pdf')
plt.savefig(SCRIPT_DIR / 'fig4_resumen_optimos.png')

print(f"\nGráficas guardadas correctamente en: {SCRIPT_DIR}")