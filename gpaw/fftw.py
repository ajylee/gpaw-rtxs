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
    """FFTW3 3d transform."""
    def __init__(self, in_R, out_R, sign, flags=FFTW_MEASURE):
        if in_R.dtype == float:
            assert sign == -1
            n0, n1, n2 = in_R.shape
            self.plan = lib.fftw_plan_dft_r2c_3d(n0, n1, n2,
                                                 in_R, out_R, flags)
        elif out_R.dtype == float:
            assert sign == 1
            n0, n1, n2 = out_R.shape
            self.plan = lib.fftw_plan_dft_c2r_3d(n0, n1, n2,
                                                 in_R, out_R, flags)
        else:
            n0, n1, n2 = in_R.shape
            self.plan = lib.fftw_plan_dft_3d(n0, n1, n2,
                                             in_R, out_R, sign, flags)
        
    def execute(self):
        lib.fftw_execute(self.plan)

    def __del__(self, lib=lib):
        lib.fftw_destroy_plan(self.plan)


class FFTPlanB:
    """Numpy fallback."""
    def __init__(self, in_R, out_R, sign, flags=None):
        self.in_R = in_R
        self.out_R = out_R
        self.sign = sign

    def execute(self):
        if self.in_R.dtype == float:
            self.out_R[:] = np.fft.rfftn(self.in_R)
        elif self.out_R.dtype == float:
            self.out_R[:] = np.fft.irfftn(self.in_R)
            self.out_R *= self.out_R.size
        elif self.sign == 1:
            self.out_R[:] = np.fft.ifftn(self.in_R)
            self.out_R *= self.out_R.size
        else:
            self.out_R[:] = np.fft.fftn(self.in_R)


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
    lib.fftw_plan_dft_r2c_3d.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=float, ndim=3),#, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=complex, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_uint]
    lib.fftw_plan_dft_c2r_3d.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=complex, ndim=3, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=float, ndim=3),#, flags='C_CONTIGUOUS'),
        ctypes.c_uint]


if __name__ == '__main__':
    shape = (32, 28, 124)
    a = empty(shape, complex)
    for Plan in [FFTPlan, FFTPlanB]:
        t0 = time.time()
        plan = Plan(a, a, -1)
        t1 = time.time()
        for i in range(50):
            a[:] = 1.3
            plan.execute()
        t2 = time.time()
        print(Plan.__name__, t1 - t0, t2 - t1)

    a = empty((32, 28, 63), complex)
    b = a.view(dtype=float)[:, :, :-2]
    for Plan in [FFTPlan, FFTPlanB]:
        t0 = time.time()
        plan = Plan(b, a, -1)
        t1 = time.time()
        for i in range(50):
            b[:] = 1.3
            plan.execute()
        t2 = time.time()
        print(Plan.__name__, t1 - t0, t2 - t1)

    c = a.copy()
    for Plan in [FFTPlan, FFTPlanB]:
        t0 = time.time()
        plan = Plan(a, b, 1)
        t1 = time.time()
        for i in range(50):
            a[:] = c
            plan.execute()
        t2 = time.time()
        print(Plan.__name__, t1 - t0, t2 - t1)
