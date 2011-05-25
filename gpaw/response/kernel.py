import numpy as np
from math import sqrt, pi
from gpaw.utilities.blas import gemmdot
from gpaw.xc import XC
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.mpi import world, rank, size


def calculate_Kc(q_c, Gvec_Gc, bcell_cv):
    """Symmetric Coulomb kernel"""
    npw = len(Gvec_Gc)
    Kc_G = np.zeros(npw)
    
    for iG in range(npw):
        qG1 = np.dot(q_c + Gvec_Gc[iG], bcell_cv)
        Kc_G[iG] = 1. / sqrt(np.dot(qG1, qG1))

    Kc_GG = 4 * pi * np.outer(Kc_G, Kc_G)

    return Kc_GG


def calculate_Kxc(gd, nt_sG, npw, Gvec_Gc, nG, vol,
                  bcell_cv, R_av, setups, D_asp):
    """LDA kernel"""

    # The soft part
    assert np.abs(nt_sG[0].shape - nG).sum() == 0

    xc = XC('LDA')
    
    fxc_sg = np.zeros_like(nt_sG)
    xc.calculate_fxc(gd, nt_sG, fxc_sg)
    fxc_g = fxc_sg[0]

    # FFT fxc(r)
    nG0 = nG[0] * nG[1] * nG[2]
    tmp_g = np.fft.fftn(fxc_g) * vol / nG0

    r_vg = gd.get_grid_point_coordinates()
    
    Kxc_GG = np.zeros((npw, npw), dtype=complex)
    for iG in range(npw):
        for jG in range(npw):
            dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
            if (nG / 2 - np.abs(dG_c) > 0).all():
                index = (dG_c + nG) % nG
                Kxc_GG[iG, jG] = tmp_g[index[0], index[1], index[2]]
            else: # not in the fft index
                dG_v = np.dot(dG_c, bcell_cv)
                dGr_g = gemmdot(dG_v, r_vg, beta=0.0) 
                Kxc_GG[iG, jG] = gd.integrate(np.exp(-1j*dGr_g)*fxc_g)

    KxcPAW_GG = np.zeros_like(Kxc_GG)
    # The PAW part
    dG_GGv = np.zeros((npw, npw, 3))
    for iG in range(npw):
        for jG in range(npw):
            dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
            dG_GGv[iG, jG] =  np.dot(dG_c, bcell_cv)

    for a, setup in enumerate(setups):
        if rank == a % size:
            rgd = setup.xc_correction.rgd
            n_qg = setup.xc_correction.n_qg
            nt_qg = setup.xc_correction.nt_qg
            nc_g = setup.xc_correction.nc_g
            nct_g = setup.xc_correction.nct_g
            Y_nL = setup.xc_correction.Y_nL
            dv_g = rgd.dv_g
        
            D_sp = D_asp[a]
            B_pqL = setup.xc_correction.B_pqL
            D_sLq = np.inner(D_sp, B_pqL.T)
            nspins = len(D_sp)
            assert nspins == 1
            
            f_sg = rgd.empty(nspins)
            ft_sg = rgd.empty(nspins)
        
            n_sLg = np.dot(D_sLq, n_qg)
            nt_sLg = np.dot(D_sLq, nt_qg)
            # Add core density
            n_sLg[:, 0] += sqrt(4 * pi) / nspins * nc_g
            nt_sLg[:, 0] += sqrt(4 * pi) / nspins * nct_g
            
            coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
        
            for n, Y_L in enumerate(Y_nL):
                w = weight_n[n]
                f_sg[:] = 0.0
                n_sg = np.dot(Y_L, n_sLg)
                xc.calculate_fxc(rgd, n_sg, f_sg)
                
                ft_sg[:] = 0.0
                nt_sg = np.dot(Y_L, nt_sLg)
                xc.calculate_fxc(rgd, nt_sg, ft_sg)
        
                coef_GGg = np.exp(-1j * np.outer(np.inner(dG_GGv, R_nv[n]),
                                                 rgd.r_g)).reshape(npw,npw,rgd.N)
                KxcPAW_GG += w * np.dot(coef_GGg, (f_sg[0]-ft_sg[0]) * dv_g) * coefatoms_GG
    world.sum(KxcPAW_GG)
    Kxc_GG += KxcPAW_GG
    
    return Kxc_GG / vol
                
