from pyscf.pbc import gto, scf
#
cell = gto.M(
    atom = '''C 0.0000 0.0000 0.0000
              C 0.8917 0.8917 0.8917''',
    a = '''0.0000 1.7834 1.7834
           1.7834 0.0000 1.7834
           1.7834 1.7834 0.0000''',
    pseudo = 'gth-pade',
    basis = 'gth-szv'
    )

kpts = cell.make_kpts([2,2,2])
kmf = scf.KRHF(cell, kpts=cell.make_kpts([2,2,2])).run()
# converged SCF energy = -10.9308552994574