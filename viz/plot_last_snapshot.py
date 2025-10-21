
# Quick & dirty plot of density snapshots written by the C++ scaffold
import glob, numpy as np, matplotlib.pyplot as plt, pandas as pd

meta = sorted(glob.glob('./data/fields_*_meta.txt'))[-1]
prefix = meta.replace('_meta.txt','')
rho = np.loadtxt(prefix + '_rho.csv', delimiter=',')
p   = np.loadtxt(prefix + '_p.csv', delimiter=',')

# Infer grid (only for toy visualization; real loader should parse meta)
with open(meta) as f:
    lines = f.read().strip().splitlines()
KV = dict([tuple(x.split('=')) for x in lines[1].replace(' ','').split(',')])
Nr = int(KV['Nr']); Nz = int(KV['Nz']); Ng = int(KV['Ng'])

rho = rho.reshape((Nr+2*Ng, Nz+2*Ng))

plt.figure()
plt.imshow(rho.T, origin='lower', aspect='auto')
plt.title('rho (cell-centered)')
plt.colorbar()
plt.show()
