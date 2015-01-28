
import numpy as np


def _seed_state_single_kpt(kth_sigma_cn, psit_nG, (n_start, n_end)):
    """ Calculate the seed state for xs for single kpt.

    Parameters
    ----------

    kth_sigma_cn: numpy.ndarray
         polarity x band. Only has bands from n_start to n_end.

    psit_nG: numpy.ndarray
         full set of psit_nG.
    """
    return np.tensordot(kth_sigma_cn, psit_nG[n_start:n_end], ([-1], [0]))


def split_sigma_cn_by_kpt(num_kpts, sigma_cn):
    """ Split sigma_cn by kpts.

    Return array has shape (num_kpts, 3, num_orbs_per_kpt)
    """
    seed_kcn = np.hsplit(sigma_cn, num_kpts)
    return seed_kcn


def seed_state(sigma_cn, wfs, (n_start, n_end)):
    """ Calculate the seed state for xs. Interface with gpaw.xas.

    Parameters
    ----------

    sigma_cn: numpy.ndarray
         polarity x band. Only has bands from n_start to n_end.

    wfs: gpaw.wavefunctions.WaveFunctions

    """
    sigma_kcn = split_sigma_cn_by_kpt(len(wfs.kpt_u), sigma_cn)

    return [_seed_state_single_kpt(kth_sigma_cn,
                               np.array(kpt.psit_nG), # convert tar array
                               (n_start, n_end))
        for kth_sigma_cn, kpt
        in zip(sigma_kcn, wfs.kpt_u)]


def distribute_seed_over_grid(grid, root_wave):
    """ Distribute seed over grid (MPI)

    Parameters
    ----------

    grid: gpaw.grid_descriptor.GridDescriptor
        Grid to distribute over.

    root_wave: numpy.ndarray
        A set of amplitudes on a grid representing the wavefunction.

    """
    local_wave = grid.empty(dtype=root_wave.dtype)
    grid.distribute(root_wave, local_wave)
    return local_wave
