from math import sqrt, pi

import numpy as np

from gpaw.xc.functional import XCFunctional
from gpaw.sphere.lebedev import Y_nL, weight_n


class LDA(XCFunctional):
    def __init__(self, kernel):
        self.kernel = kernel
        XCFunctional.__init__(self, kernel.name)
        self.type = kernel.type

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        if gd is not self.gd:
            self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        self.calculate_lda(e_g, n_sg, v_sg)
        return gd.integrate(e_g)

    def calculate_lda(self, e_g, n_sg, v_sg):
        self.kernel.calculate(e_g, n_sg, v_sg)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        c = setup.xc_correction
        if c is None:
            return 0.0
        
        rgd = c.rgd
        nspins = len(D_sp)
        
        if addcoredensity:
            nc0_sg = rgd.empty(nspins)
            nct0_sg = rgd.empty(nspins)
            nc0_sg[:] = sqrt(4 * pi) / nspins * c.nc_g
            nct0_sg[:] = sqrt(4 * pi) / nspins * c.nct_g
            if c.nc_corehole_g is not None and nspins == 2:
                nc0_sg[0] -= 0.5 * sqrt(4 * pi) * c.nc_corehole_g
                nc0_sg[1] += 0.5 * sqrt(4 * pi) * c.nc_corehole_g
        else:
            nc0_sg = 0
            nct0_sg = 0
        
        D_sLq = np.inner(D_sp, c.B_pqL.T)

        e, dEdD_sqL = self.calculate_radial_expansion(rgd, D_sLq, c.n_qg,
                                                      nc0_sg)
        et, dEtdD_sqL = self.calculate_radial_expansion(rgd, D_sLq, c.nt_qg,
                                                        nct0_sg)

        if dEdD_sp is not None:
            dEdD_sp += np.inner((dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                                c.B_pqL.reshape((len(c.B_pqL), -1)))
            
        if addcoredensity:
            return e - et - c.Exc0
        else:
            return e - et

    def calculate_radial_expansion(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg

        dEdD_sqL = np.zeros_like(np.transpose(D_sLq, (0, 2, 1)))

        Lmax = n_sLg.shape[1]
        
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]

            e_g, dedn_sg = self.calculate_radial(rgd, n_sLg, Y_L)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            E += w * rgd.integrate(e_g)

        return E, dEdD_sqL

    def calculate_radial(self, rgd, n_sLg, Y_L):
        nspins = len(n_sLg)

        n_sg = np.dot(Y_L, n_sLg)
        e_g = rgd.empty()
        dedn_sg = rgd.zeros(nspins)

        self.kernel.calculate(e_g, n_sg, dedn_sg)

        return e_g, dedn_sg

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        if e_g is None:
            e_g = rgd.empty()
        e_g[:], dedn_sg = self.calculate_radial(rgd, n_sg[:, np.newaxis],
                                                [1.0])
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)

    def calculate_fxc(self, gd, n_sg, f_sg):
        if gd is not self.gd:
            self.set_grid_descriptor(gd)

        assert len(n_sg) == 1
        assert n_sg.shape == f_sg.shape
        assert n_sg.flags.contiguous and n_sg.dtype == float
        assert f_sg.flags.contiguous and f_sg.dtype == float
        self.kernel.xc.calculate_fxc_spinpaired(n_sg.ravel(), f_sg)


class PurePythonLDAKernel:
    def __init__(self):
        self.name = 'LDA'
        self.type = 'LDA'
        
    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        assert len(n_sg) == 1
        lda(e_g, n_sg[0], dedn_sg[0])


def lda(e, n, v):
    C0I = 0.238732414637843
    C1 = -0.45816529328314287  
    n[n < 1e-20] = 1e-40
    rs = (C0I / n)**(1 / 3.0)
    ex = C1 / rs
    dexdrs = -ex / rs;
    ec, decdrs = G(rs**0.5)
    e[:] = n * (ex + ec)
    v += ex + ec - rs * (dexdrs + decdrs) / 3.0
    

def G(rtrs):
    A = 0.031091
    alpha1 = 0.21370
    beta1, beta2, beta3, beta4 = 7.5957, 3.5876, 1.6382, 0.49294
    Q0 = -2.0 * A * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * A * rtrs * (beta1 + 
                           rtrs * (beta2 + 
                                   rtrs * (beta3 + 
                                           rtrs * beta4)))
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    dQ1drs = A * (beta1 / rtrs + 2.0 * beta2 +
                  rtrs * (3.0 * beta3 + 4.0 * beta4 * rtrs))
    dGdrs = -2.0 * A * alpha1 * G1 / Q0 - Q0 * dQ1drs / (Q1 * (Q1 + 1.0))
    return G1, dGdrs
