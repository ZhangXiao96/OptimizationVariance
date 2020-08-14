import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# sb.set()

model_name = 'ResNet18'
data_name = 'cifar10'
noise = 0.2
opt = 'adam'
lr = 0.0001
test_id = 0
runs = 'runs/noise_{}_opt_{}_lr_{}'.format(noise, opt, lr)

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
ax.set_xlim(0, 250)
ax2 = ax.twinx()

ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('acc', fontsize=14)
ax2.set_ylabel('OV', fontsize=14)

test_dir = os.path.join(runs, data_name, model_name.lower(), '{}'.format(test_id), 'log')

test_path = os.path.join(test_dir, 'test_acc.csv')
test_results = pd.read_csv(test_path)
test_epoch = test_results['steps']
test_acc = test_results['values']

ov_path = os.path.join(test_dir, 'optimization_var.csv')
ov_results = pd.read_csv(ov_path)
ov_epoch= ov_results['steps']
ov = ov_results['values']

ax.plot(test_epoch, test_acc, color='C0', linestyle='-')
ax2.plot(ov_epoch, ov, color='C1', linestyle='-')

# plt.xscale('log')

ax.plot([-2, -1], [0, 0], color='C0', linestyle='-', label='acc')
ax.plot([-2, -1], [0, 0], color='C1', linestyle='-', label='OV')

ax.legend(loc='lower right', fontsize=12)
plt.title(model_name, fontsize=14)
plt.tight_layout()
plt.show()
