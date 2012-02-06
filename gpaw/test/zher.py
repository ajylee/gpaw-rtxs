from gpaw.utilities.blas import zher
import numpy as np

alpha = 0.5
x = np.random.rand(3) + 1j * np.random.rand(3)
a = np.random.rand(9).reshape(3,3) + np.random.rand(9).reshape(3,3) * 1j

# make a hermitian
for i in range(3):
    for j in range(3):
        a[i,j] = a[j,i].conj()
    a[i,i] = np.real(a[i,i])

b = alpha * np.outer(x.conj(), x) + a
zher(alpha, x, a)

for i in range(3):
    for j in range(i,3):
        a[j,i] = a[i,j].conj()


assert np.abs(b-a).sum() < 1e-14
