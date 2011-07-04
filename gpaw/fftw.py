"""Python wrapper for FFTW3 library."""

import time

import numpy as np

try:
    import ctypes
    lib = ctypes.CDLL('libfftw3.so')
except (ImportError, OSError):
    lib = None  # Use plan B


FFTW_ESTIMATE = 64
FFTW_MEASURE = 0
FFTW_PATIENT = 32
FFTW_EXHAUSTIVE = 8


class FFTPlan:
    """FFTW3 c2c inplace 3d teransform."""
    def __init__(self, tmp_R, sign, flags=FFTW_MEASURE):
        n0, n1, n2 = tmp_R.shape
        self.plan = lib.fftw_plan_dft_3d(n0, n1, n2,
                                         tmp_R, tmp_R, sign, flags)
        
    def execute(self):
        lib.fftw_execute(self.plan)

    def __del__(self, lib=lib):
        lib.fftw_destroy_plan(self.plan)


class FFTPlanB:
    """Numpy fallback."""
    def __init__(self, tmp_R, sign, flags=0):
        self.tmp_R = tmp_R
        self.sign = sign
        self.tplan = 0

    def execute(self):
        if self.sign == 1:
            self.tmp_R[:] = np.fft.ifftn(self.tmp_R)
            self.tmp_R *= self.tmp_R.size
        else:
            self.tmp_R[:] = np.fft.fftn(self.tmp_R)


def empty(shape, dtype=float):
    """numpy.empty() equivalent with 16 byte allignment."""
    assert dtype == complex
    N = np.prod(shape)
    a = np.empty(2 * N + 1)
    offset = (a.ctypes.data % 16) // 8
    a = a[offset:2 * N + offset].view(complex)
    a.shape = shape
    return a


if lib is None:
    FFTPlan = FFTPlanB
else:
    lib.fftw_plan_dft_3d.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=complex, ndim=3, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=complex, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_uint]


if __name__ == '__main__':
    shape = (32, 28, 124)
    a = empty(shape, complex)
    for Plan in [FFTPlan, FFTPlanB]:
        t0 = time.time()
        plan = Plan(a, -1)
        t1 = time.time()
        for i in range(50):
            a[:] = 1.3
            plan.execute()
        t2 = time.time()
        print(Plan.__name__, t1 - t0, t2 - t1)
