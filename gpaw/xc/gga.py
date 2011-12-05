from math import pi

import numpy as np

from gpaw.xc.lda import LDA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv


class GGA(LDA):
    def set_grid_descriptor(self, gd):
        LDA.set_grid_descriptor(self, gd)
        self.grad_v = [Gradient(gd, v).apply for v in range(3)]

    def calculate_lda(self, e_g, n_sg, v_sg):
        nspins = len(n_sg)
        gradn_svg = self.gd.empty((nspins, 3))
        sigma_xg = self.gd.zeros(nspins * 2 - 1)
        dedsigma_xg = self.gd.empty(nspins * 2 - 1)
        for v in range(3):
            for s in range(nspins):
                self.grad_v[v](n_sg[s], gradn_svg[s, v])
                axpy(1.0, gradn_svg[s, v]**2, sigma_xg[2 * s])
            if nspins == 2:
                axpy(1.0, gradn_svg[0, v] * gradn_svg[1, v], sigma_xg[1])
        self.calculate_gga(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        vv_g = sigma_xg[0]
        for v in range(3):
            for s in range(nspins):
                self.grad_v[v](dedsigma_xg[2 * s] * gradn_svg[s, v], vv_g)
                axpy(-2.0, vv_g, v_sg[s])
                if nspins == 2:
                    self.grad_v[v](dedsigma_xg[1] * gradn_svg[s, v], vv_g)
                    axpy(-1.0, vv_g, v_sg[1 - s])
                    # TODO: can the number of gradient evaluations be reduced?

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        
    def calculate_radial_expansion(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg

        dndr_sLg = np.empty_like(n_sLg)
        for n_Lg, dndr_Lg in zip(n_sLg, dndr_sLg):
            for n_g, dndr_g in zip(n_Lg, dndr_Lg):
                rgd.derivative(n_g, dndr_g)

        nspins, Lmax, nq = D_sLq.shape
        dEdD_sqL = np.zeros((nspins, nq, Lmax))
        
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]
            rnablaY_Lv = rnablaY_nLv[n, :Lmax]
            e_g, dedn_sg, b_vsg, dedsigma_xg = \
                 self.calculate_radial(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            dedsigma_xg *= rgd.dr_g
            B_vsg = dedsigma_xg[::2] * b_vsg
            if nspins == 2:
                B_vsg += 0.5 * dedsigma_xg[1] * b_vsg[:, ::-1]
            B_vsq = np.dot(B_vsg, n_qg.T)
            dEdD_sqL += 8 * pi * w * np.inner(rnablaY_Lv, B_vsq.T).T
            E += w * rgd.integrate(e_g)

        return E, dEdD_sqL

    def calculate_radial(self, rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv):
        nspins = len(n_sLg)

        n_sg = np.dot(Y_L, n_sLg)

        a_sg = np.dot(Y_L, dndr_sLg)
        b_vsg = np.dot(rnablaY_Lv.T, n_sLg)

        sigma_xg = rgd.empty(2 * nspins - 1)
        sigma_xg[::2] = (b_vsg**2).sum(0)
        if nspins == 2:
            sigma_xg[1] = (b_vsg[:, 0] * b_vsg[:, 1]).sum(0)
        sigma_xg[:, 1:] /= rgd.r_g[1:]**2
        sigma_xg[:, 0] = sigma_xg[:, 1]
        sigma_xg[::2] += a_sg**2
        if nspins == 2:
            sigma_xg[1] += a_sg[0] * a_sg[1]

        e_g = rgd.empty()
        dedn_sg = rgd.zeros(nspins)
        dedsigma_xg = rgd.zeros(2 * nspins - 1)

        self.calculate_gga_radial(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        
        vv_sg = sigma_xg[:nspins]  # reuse array
        for s in range(nspins):
            rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[2 * s] * a_sg[s],
                            vv_sg[s])
        if nspins == 2:
            v_g = sigma_xg[2]
            rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * a_sg[1], v_g)
            vv_sg[0] -= v_g
            rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * a_sg[0], v_g)
            vv_sg[1] -= v_g

        vv_sg[:, 1:] /= rgd.dv_g[1:]
        vv_sg[:, 0] = vv_sg[:, 1]
        
        return e_g, dedn_sg + vv_sg, b_vsg, dedsigma_xg

    calculate_gga_radial = calculate_gga

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        dndr_sg = np.empty_like(n_sg)
        for n_g, dndr_g in zip(n_sg, dndr_sg):
            rgd.derivative(n_g, dndr_g)
        if e_g is None:
            e_g = rgd.empty()
        e_g[:], dedn_sg = self.calculate_radial(rgd, n_sg[:, np.newaxis],
                                                [1.0],
                                                dndr_sg[:, np.newaxis],
                                                np.zeros((1, 3)))[:2]
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)
