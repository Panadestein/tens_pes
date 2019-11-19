import sys
import numpy as np
import matplotlib.pyplot as plt

# Taking argument from command line

DIR = sys.argv[1]

# Defining variables and parsing output files

with open(DIR + '/table_energies', 'r') as f1:
    e_ab, e_srp = np.loadtxt(f1, unpack=True)

index = np.argsort(e_ab)
e_ab_sorted = np.sort(e_ab)
e_srp_sorted = e_srp[index]

rms = []
for i in np.arange(len(e_ab)):
    val = np.sqrt(((e_ab_sorted[0:i] - e_srp_sorted[0:i]) ** 2).mean())
    rms.append(val)
rms = np.array(rms)

# Plot the energies

fig, ax = plt.subplots()
plt.ylabel(r'$E_{ref} \, (\hbar \omega)$', fontsize=16)
plt.xlabel(r'$E_{sop} \, (\hbar \omega)$', fontsize=16)
ax.plot(e_ab, e_srp, 'r.', markersize=2)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.tight_layout()
plt.show()


plt.plot(e_ab_sorted, rms)
plt.xlabel(r'$E_{ref} \, (\hbar \omega)$')
plt.ylabel(r'RMSE')
plt.tight_layout()
plt.show()
