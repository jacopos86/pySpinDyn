import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import matplotlib.pyplot as plt
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath
#
crystal = ase.build.bulk('Si', 'diamond', a=5.43102)
V = crystal.get_volume()
# pyscf cell
unit_cell = pbcgto.Cell()
unit_cell.atom = pyscf_ase.ase_atoms_to_pyscf(crystal)
unit_cell.a = crystal.cell
unit_cell.basis = 'gth-szv'
unit_cell.pseudo = 'gth-pade'
unit_cell.verbose = 5
unit_cell.build(None,None)
points = special_points['fcc']
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], crystal.cell, npoints=50)
band_kpts = unit_cell.get_abs_kpts(band_kpts)
#
# band structure from Gamma sampling
#
mf = pbcdft.KRKS(unit_cell)
print(mf.kernel())
#
e_kn = mf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn:
    vb_k = en[unit_cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn = [en - vbmax for en in e_kn]
#
# band structure from 222 k-pt sampling
#
kmf = pbcdft.KRKS(unit_cell, unit_cell.make_kpts([2,2,2]))
print(kmf.kernel())
#
e_kn_2 = kmf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[unit_cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]
#
au2ev = 27.21139
emin = -1*au2ev
emax = 1*au2ev
#
plt.figure(figsize=(5, 6))
nbands = unit_cell.nao_nr()
for n in range(nbands):
    plt.plot(kpath, [e[n]*au2ev for e in e_kn], color='#87CEEB')
    plt.plot(kpath, [e[n]*au2ev for e in e_kn_2], color='#4169E1')
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')
plt.show()