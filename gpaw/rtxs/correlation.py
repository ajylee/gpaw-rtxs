import numpy as np
from numpy.linalg import det

def _apply_overlap_single_kpt(wfs, kpt, a_xG):
    """Apply the all electron overlap operator.

    Caveat: I am only able to get this to work with TD wfs. Otherwise
    wfs.overlap fails. If you are doing FS, then convert it to TD first.

    Parameters
    ----------

    a_xG: numpy.ndarray
        array of wavefunctions to apply overlap operator to.
        a_xG.shape = (nbands, xsize - 1, ysize-1, zsize-1)

    """
    lapped =  a_xG.copy()
    wfs.overlap.apply(a_xG = a_xG, b_xG = lapped,
                      wfs = wfs, kpt = kpt, calculate_P_ani=True)
    return lapped

def apply_overlap(wfs, list_of_a_xG):
    """See _apply_overlap_single_kpt

    Parameters
    ----------

    list_of_a_xG: list of numpy.ndarray
        Indexed by kpt. Each element is an array of wavefunctions to apply
        overlap operator to. a_xG.shape = (nkpts, nbands, xsize - 1, ysize-1, zsize-1)

    """
    return np.array([_apply_overlap_single_kpt(wfs, kpt, a_xG)
                 for kpt, a_xG in zip(wfs.kpt_u, list_of_a_xG)])



def _up_vectors(kpt, unocc_photoelectron_idx=None):
    """Valence up electrons, plus one optional unoccupied photoelectron vector"""
    valence_idx = [nn for nn in xrange(len(kpt.psit_nG))
                   if kpt.f_n[nn] > .9]

    idx = valence_idx + ([unocc_photoelectron_idx] if unocc_photoelectron_idx else [])

    return kpt.psit_nG[idx]


def _down_vectors(kpt):
    return np.array([kpt.psit_nG[nn] for nn in xrange(len(kpt.psit_nG))
                 if kpt.f_n[nn] > 1.9])


def _many_body_inner_product(integrate, unocc_photoelectron_idx,
                             wfs, bra_kpt_u, ket_kpt_u):

    weighted_inner_products_u = np.zeros(len(bra_kpt_u), dtype=complex)

    for uu in range(len(bra_kpt_u)):
        bra_kpt = bra_kpt_u[uu]
        ket_kpt = ket_kpt_u[uu]

        ## NOTE: GPAW's grid_descriptor integrate method includes the
        ## appropriate conjugation.
        up_product = det(integrate(
            _up_vectors(bra_kpt, unocc_photoelectron_idx),
            _apply_overlap_single_kpt(wfs, ket_kpt,
                                      _up_vectors(ket_kpt, unocc_photoelectron_idx))))

        down_product = det(integrate(
            _down_vectors(bra_kpt),
            _apply_overlap_single_kpt(wfs, ket_kpt, _down_vectors(ket_kpt))))

        #weighted_inner_products_u[uu] = bra_kpt.weight * up_product * down_product
        weighted_inner_products_u[uu] = up_product * down_product

    return sum(weighted_inner_products_u)


def _one_body_inner_product(integrate, idx,
                             wfs, bra_kpt_u, ket_kpt_u):

    weighted_inner_products_u = np.zeros(len(bra_kpt_u), dtype=complex)

    for uu in range(len(bra_kpt_u)):
        bra_kpt = bra_kpt_u[uu]
        ket_kpt = ket_kpt_u[uu]

        product = det(integrate(
            bra_kpt.psit_nG[[idx]],
            _apply_overlap_single_kpt(wfs, ket_kpt,
                                      ket_kpt.psit_nG[[idx]])))

        #weighted_inner_products_u[uu] = bra_kpt.weight * product
        weighted_inner_products_u[uu] = product

    return sum(weighted_inner_products_u)


class GuttedKPoint:
    def __init__(self, kpt):
        self.f_n = kpt.f_n.copy()
        self.psit_nG = kpt.psit_nG.copy()
        self.q = kpt.q
        self.weight = kpt.weight


class OneBodyCorrelation(object):
    """Calculates autocorrelation for each cycle."""

    ## architecturally, this class more closely resembles ManyBodyCorrelation

    def __init__(self, wfs, photoelectron_indices, use_paw=True):

        self.seed_kpt_u = [GuttedKPoint(kpt) for kpt in wfs.kpt_u]
        self.photoelectron_indices = photoelectron_indices

    def __call__(self, new_wfs):
        #integrate = np.frompyfunc(new_wfs.gd.integrate, 2, 1)
        integrate = new_wfs.gd.integrate

        ans = [_one_body_inner_product(integrate, _idx,
                                        new_wfs, self.seed_kpt_u, new_wfs.kpt_u,)
               for _idx in self.photoelectron_indices]

        return ans


class ManyBodyCorrelation(object):
    def __init__(self, wfs, unocc_photoelectron_indices=None):
        """Calculates autocorrelation for each cycle.
        """

        self.seed_kpt_u = [GuttedKPoint(kpt) for kpt in wfs.kpt_u]
        self.unocc_photoelectron_indices = tuple(unocc_photoelectron_indices)

    def __call__(self, new_wfs):
        #integrate = np.frompyfunc(new_wfs.gd.integrate, 2, 1)
        integrate = new_wfs.gd.integrate

        if self.unocc_photoelectron_indices:

            ans = [_many_body_inner_product(integrate, unocc_photoelectron_idx,
                                            new_wfs, self.seed_kpt_u, new_wfs.kpt_u,)
                   for unocc_photoelectron_idx in self.unocc_photoelectron_indices]
        else:
            ans = [_many_body_inner_product(integrate, None,
                                            new_wfs, self.seed_kpt_u, new_wfs.kpt_u,)]

        return ans
