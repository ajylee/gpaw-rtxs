import math as ma
import numpy as np
from .correlation import _apply_overlap_single_kpt


def norms(wfs, kpt, a_xG):

    lapped_x = _apply_overlap_single_kpt(wfs=wfs, kpt=kpt,
                                         a_xG=a_xG)

    return np.array([
        ma.sqrt(abs(wfs.gd.integrate(a_xG[xx].conjugate(), lapped_x[xx])))
        for xx in xrange(len(a_xG))])
