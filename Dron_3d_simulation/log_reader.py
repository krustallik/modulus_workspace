#!/usr/bin/env python3
import re
import csv
import matplotlib
# Якщо вам не потрібне показування вікна — виберіть бекенд 'Agg'
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = 'log/log.txt'
BIN_SIZE = 5000
OUT_CSV = 'loss_avg.csv'
OUT_PNG = 'loss_avg_plot.png'

# Регекс для парсингу
pattern = re.compile(r'\[step:\s*(\d+)\].*?loss:\s*([\d\.e\+\-]+)')

# Збираємо всі пари (step, loss)
data = []
with open(LOG_PATH, 'r') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            data.append((step, loss))

# Групуємо по вікнах
sums = {}
counts = {}
for step, loss in data:
    bin_start = (step // BIN_SIZE) * BIN_SIZE
    sums.setdefault(bin_start, 0.0)
    counts.setdefault(bin_start, 0)
    sums[bin_start] += loss
    counts[bin_start] += 1

# Обчислюємо середні та готуємо списки для запису/плоту
bins = sorted(sums.keys())
avg_losses = []
with open(OUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['step_bin_start', 'avg_loss'])
    for b in bins:
        avg = sums[b] / counts[b]
        avg_losses.append(avg)
        writer.writerow([b, f'{avg:.6f}'])
print(f"Середні loss по кожних {BIN_SIZE} кроках записані у {OUT_CSV}")

# Малюємо графік
plt.figure(figsize=(10, 6))
plt.plot(bins, avg_losses, marker='o', linestyle='-')
plt.title(f'Average Loss per {BIN_SIZE} Steps')
plt.xlabel('Step (bin start)')
plt.ylabel('Average Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Графік середніх loss збережено у файлі {OUT_PNG}")