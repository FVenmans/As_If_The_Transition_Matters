import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Parameters
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

LOG_DIR = Parameters.LOG_DIR
versions = {
    #1: 'RRA=1.35',
    2: 'Temp and capital unc.',
    3: 'Only temp unc.',
    4: 'Only capital unc.',
    #5: 'No unc.',
    #6: 'RRA=3',
}
years = [2030, 2050, 2100]
fsize = (12, 9)
colors = {
    1: 'tab:blue',
    2: 'tab:orange',
    3: 'tab:green',
    4: 'tab:red',
    5: 'tab:purple',
    6: 'magenta',
}
variables = {
    'T':   'Temperature (°C)',
    'kc':  'Clean capital',
    'kd':  'Dirty capital',
    'P':   'GHG Emissions (GtCO2e)',
    'SCC': 'Social Cost of Carbon',
}

# Font sizes (larger by 50% because 3 graphs per line)
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24

for varname, ylabel in variables.items():
    fig, ax = plt.subplots(figsize=fsize)
    n_versions = len(versions)
    n_years = len(years)
    group_width = 0.8
    box_width = group_width / n_versions
    for vidx, (v, label) in enumerate(versions.items()):
        df = pd.read_csv(LOG_DIR + f'/{varname}_boxplot_v{v}.csv', index_col=0)
        for yidx, year in enumerate(years):
            col = str(year)
            p5, p25, p50, p75, p95 = df[col].values
            x = yidx + (vidx - n_versions / 2 + 0.5) * box_width
            ax.bar(x, p75 - p25, bottom=p25, width=box_width * 0.9, color=colors[v], alpha=0.7)
            ax.plot([x - box_width*0.45, x + box_width*0.45], [p50, p50], color='black', linewidth=1.5)
            ax.plot([x, x], [p5,  p25], color=colors[v], linewidth=1)
            ax.plot([x, x], [p75, p95], color=colors[v], linewidth=1)
            ax.plot([x - box_width*0.2, x + box_width*0.2], [p5,  p5],  color=colors[v], linewidth=1)
            ax.plot([x - box_width*0.2, x + box_width*0.2], [p95, p95], color=colors[v], linewidth=1)
    ax.set_xticks(range(n_years))
    ax.set_xticklabels(years)
    ax.set_ylabel(ylabel)
    legend_patches = [mpatches.Patch(color=colors[v], label=label)
                      for v, label in versions.items()]
    if varname == 'SCC':
        ax.legend(handles=legend_patches, loc='upper left')
    plt.tight_layout()
    plt.savefig(LOG_DIR + f'/boxplot_{varname}.pdf')
    plt.close()