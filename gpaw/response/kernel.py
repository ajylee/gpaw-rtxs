import numpy as np
from math import sqrt, pi, sin, cos, exp
from gpaw.utilities.blas import gemmdot
from gpaw.xc import XC
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.mpi import world, rank, size

def v3D_Coulomb(qG):
    """Coulomb Potential in the 3D Periodic Case
    Periodic calculation, no cutoff.
    v3D = 4 pi / G^2"""
    return 1. / np.dot(qG,qG)

def v2D_Coulomb(qG,G_p,G_n,R):
    """ 2D Periodic Case
    Slab/Surface/Layer calculation, cutoff in G_n direction.
    v2D = 4 pi/G^2 * [1 + exp(-G_p R)*[(G_n/G_p)*sin(G_n R) - cos(G_n R)]
    """

    G_nR = np.dot(G_n,qG)*R
    G_pR = sqrt(np.dot(G_p,qG*qG))*R
    return 1. / np.dot(qG,qG) * (1 + exp(-G_pR)*((G_nR/G_pR)*sin(G_nR) - cos(G_nR)))

def v1D_Coulomb(qG,G_p,G_n,R):
    """ 1D Periodic Case
    Nanotube/Nanowire/Atomic Chain calculation, cutoff in G_n direction.
     v1D = 4 pi/G^2 * [1 + G_n R J_1(G_n R) K_0(|G_p|R)
     		    - |G_p| R J_0(G_n R) K_1(|G_p|R)]
    """        
    from scipy.special import j1,k0,j0,k1
    
    G_nR = sqrt(np.dot(G_n,qG*qG))*R
    G_pR = abs(np.dot(G_p,qG))*R
    return 1. / np.dot(qG,qG) * (1 + G_nR * j1(G_nR) * k0(G_pR) - G_pR * j0(G_nR) * k1(G_pR))

def v0D_Coulomb(qG,R):
    """ 0D Non-Periodic Case
    Isolated System/Molecule calculation, spherical cutoff.
    v0D = 4 pi/G^2 * [1 - cos(G R)
    """

    qG2 = np.dot(qG,qG)
    return 1. / qG2 * (1 - cos(sqrt(qG2)*R))


def calculate_Kc(q_c,
                 Gvec_Gc,
                 acell_cv, bcell_cv,
                 pbc,
                 optical_limit,
                 vcut=None,
                 density_cut=None):
    """Symmetric Coulomb kernel"""
    npw = len(Gvec_Gc)
    Kc_G = np.zeros(npw)

    # get cutoff parameters
    G_p = np.array(pbc, float)
    # Normal Direction
    G_n = np.array([1,1,1])-G_p
        
    if vcut is None:
        pass
    elif vcut == '2D':
        if G_n.sum() < 1e-8: # default dir is z
            G_n = np.array([0,0,1])
            G_p = np.array([1,1,0])
        acell_n = acell_cv*G_n
        R = max(acell_n[0,0],acell_n[1,1],acell_n[2,2])/2.
    elif vcut == '1D':
        # R is the radius of the cylinder containing the cell.
        if G_n.sum() < 1e-8:
            raise ValueError('Check boundary condition ! ')
        acell_n = acell_cv*G_n
        R = max(acell_n[0,0],acell_n[1,1],acell_n[2,2])/2.            
    elif vcut == '0D':
        # R is the minimum radius of a sphere containing the cell.
        acell_n = acell_cv
        R = min(acell_n[0,0],acell_n[1,1],acell_n[2,2])/2.
    else:
        XXX
        NotImplemented

    # calculate coulomb kernel
    for iG in range(npw):
        qG = np.dot(q_c + Gvec_Gc[iG], bcell_cv)
        
        if vcut is None:
            Kc_G[iG] = sqrt(v3D_Coulomb(qG))
        elif vcut == '2D':
            Kc_G[iG] = sqrt(v2D_Coulomb(qG,G_p,G_n,R))
        elif vcut == '1D':
            Kc_G[iG] = sqrt(v1D_Coulomb(qG,G_p,G_n,R))
        elif vcut == '0D':
            Kc_G[iG] = sqrt(v0D_Coulomb(qG,R))
        else:
            NotImplemented
            
    if optical_limit:
        q_v = np.dot(q_c, bcell_cv)
        Kc_G[0] = sqrt(1. / np.dot(q_v,q_v))
            
    Kc_GG = 4 * pi * np.outer(Kc_G, Kc_G)

    return Kc_GG


def calculate_Kxc(gd, nt_sG, npw, Gvec_Gc, nG, vol,
                  bcell_cv, R_av, setups, D_asp, functional='ALDA',
                  density_cut=None):
    """LDA kernel"""

    # The soft part
    #assert np.abs(nt_sG[0].shape - nG).sum() == 0
    if functional == 'ALDA_X':
        x_only = True
    else:
        assert len(nt_sG) == 1
        x_only = False
    if x_only:
        A_x = -(3/4.) * (3/np.pi)**(1/3.)
        if len(nt_sG) == 1:
            fxc_sg = (4 / 9.) * A_x * nt_sG**(-2/3.)
        else:
            fxc_sg = 2 * (4 / 9.) * A_x * (2*nt_sG)**(-2/3.)
    else:
        fxc_sg = np.zeros_like(nt_sG)
        xc = XC(functional[1:])
        xc.calculate_fxc(gd, nt_sG, fxc_sg)
        
    if density_cut is not None:
        fxc_sg[np.where(nt_sG*len(nt_sG) < density_cut)] = 0.0
        
    # FFT fxc(r)
    nG0 = nG[0] * nG[1] * nG[2]
    tmp_sg = [np.fft.fftn(fxc_sg[s]) * vol / nG0 for s in range(len(nt_sG))]

    r_vg = gd.get_grid_point_coordinates()
    Kxc_sGG = np.zeros((len(fxc_sg), npw, npw), dtype=complex)
    for s in range(len(fxc_sg)):
        for iG in range(npw):
            for jG in range(npw):
                dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                if (nG / 2 - np.abs(dG_c) > 0).all():
                    index = (dG_c + nG) % nG
                    Kxc_sGG[s, iG, jG] = tmp_sg[s][index[0], index[1], index[2]]
                else: # not in the fft index
                    dG_v = np.dot(dG_c, bcell_cv)
                    dGr_g = gemmdot(dG_v, r_vg, beta=0.0) 
                    Kxc_sGG[s, iG, jG] = gd.integrate(np.exp(-1j*dGr_g)*fxc_sg[s])

    # The PAW part
    KxcPAW_sGG = np.zeros_like(Kxc_sGG)
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
                if x_only:
                    f_sg = nspins * (4 / 9.) * A_x * (nspins*n_sg)**(-2/3.)
                else:
                    xc.calculate_fxc(rgd, n_sg, f_sg)
                
                ft_sg[:] = 0.0
                nt_sg = np.dot(Y_L, nt_sLg)
                if x_only:
                    ft_sg = nspins * (4 / 9.) * A_x * (nspins*nt_sg)**(-2/3.)
                else:
                    xc.calculate_fxc(rgd, nt_sg, ft_sg)
                for i in range(len(rgd.r_g)):
                    coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n]) * rgd.r_g[i])
                    for s in range(len(f_sg)):
                        KxcPAW_sGG[s] += w * np.dot(coef_GG,
                                                    (f_sg[s,i]-ft_sg[s,i]) * dv_g[i]) \
                                                    * coefatoms_GG

    world.sum(KxcPAW_sGG)
    Kxc_sGG += KxcPAW_sGG

    return Kxc_sGG / vol
                

