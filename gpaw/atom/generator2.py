#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, exp, sqrt, log

import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from ase.utils import prnt
from ase.units import Hartree

from gpaw.utilities import erf
from gpaw.spline import Spline
from gpaw.setup import BaseSetup
from gpaw.version import version
from gpaw.basis_data import Basis
from gpaw.setup_data import SetupData
from gpaw.atom.configurations import configurations
from gpaw.utilities.lapack import general_diagonalize
from gpaw.atom.aeatom import AllElectronAtom, Channel, parse_ld_str, colors, \
    GaussianBasis


parameters = {
'H':  ('1s,s,p', 1.0),
'He': ('1s,s,p', 1.0),
'Li': ('2s,s,2p', 2.4),
'Be': ('2s,s,2p', 2.0),
'B':  ('2s,s,2p,p,d', 1.4),
'C':  ('2s,s,2p,p,d', 1.3),
'N':  ('2s,s,2p,p,d', 1.3),
'O':  ('2s,s,2p,p,d', 1.5),
'F':  ('2s,s,2p,p,d', 1.4),
'Ne': ('2s,s,2p,p,d', 1.8),
'Na': ('3s,s,p', 2.8),
'Mg': ('3s,s,p', 2.8),
'Al': ('3s,s,3p,p,d', 2.2),
'Si': ('3s,s,3p,p,d', 2.4),
'P':  ('3s,s,3p,p,d', 1.9),
'S':  ('3s,s,3p,p,d', 2.2),
'Cl': ('3s,s,3p,p,d', 2.1),
'Ar': ('3s,s,3p,p,d', 2.0),
'K':  ('3s,4s,3p,4p,d', 2.4),
'Ca': ('3s,4s,3p,4p,3d', 2.5),
'Sc': ('3s,4s,3p,4p,3d,d', 2.8, 'f'),
'Ti': ('3s,4s,3p,4p,3d,d', 2.7, 'f'),
'V':  ('3s,4s,3p,4p,3d,d', 2.7, 'f'),
'Cr': ('3s,4s,3p,4p,3d,d', 2.7, 'f'),
'Mn': ('4s,s,4p,p,3d,d', 2.8),
'Fe': ('4s,s,4p,p,3d,d', 2.8),
'Co': ('4s,s,4p,p,3d,d', 2.7),
'Ni': ('4s,s,4p,p,3d,d', 2.7),
'Cu': ('4s,s,4p,p,3d,d', 2.5),
'Zn': ('4s,s,4p,p,3d,d', 2.8),
'Ga': ('4s,s,4p,p,3d,d', 2.6),
'Ge': ('4s,s,4p,p,d', 2.7),
'As': ('4s,s,4p,p,d', 2.7),
'Se': ('4s,s,4p,p,d', 2.7),
'Br': ('4s,s,4p,p,d', 2.7),
'Kr': ('4s,s,4p,p,d', 2.4),
'Rb': ('4s,5s,4p,5p,d', 2.7),
'Sr': ('4s,5s,4p,5p,d', 2.9),
'Y':  ('5s,s,?p,p,4d,d', 3.0),
'Zr': ('5s,s,?p,p,4d,d', 2.9),
'Nb': ('5s,s,?p,p,4d,d', 2.9),
'Mo': ('5s,s,?p,p,4d,d', 2.9),
'Ru': ('4s,5s,4p,5p,4d,d', 2.9, 'f'),
'Rh': ('5s,s,?p,p,4d,d', 2.9),
'Pd': ('?s,s,?p,p,4d,d', 2.8),
'Ag': ('5s,s,5p,p,4d,d', 3.1),
'Cd': ('5s,s,?p,p,4d,d', 2.7),
'In': ('5s,s,5p,p,4d,d', 2.7),
'Sn': ('5s,s,5p,p,d', 2.8),
'Sb': ('5s,s,5p,p,d', 2.7),
'Te': ('5s,s,5p,p,d', 2.7),
'I':  ('5s,s,5p,p,d', 2.7),
'Xe': ('5s,s,5p,p,d', 2.8),
'Cs': ('5s,6s,5p,p,d', 3.2),
'Ba': ('6s,s,5p,p,d', 3.0),
'La': ('6s,s,5p,p,5d,d', 3.1),
'Ce': ('5s,6s,5p,6p,5d,d', 2.8, 'f', 1.5),
'Lu': ('6s,s,?p,p,5d,d,4f,f', 3.4),
'Hf': ('6s,s,?p,p,5d,d,4f,f', 3.1),
'Ta': ('6s,s,?p,p,5d,d', 3.0),
'W':  ('6s,s,?p,p,5d,d', 3.0),
'Re': ('6s,s,?p,p,5d,d', 3.0),
'Os': ('6s,s,?p,p,5d,d', 3.0),
'Ir': ('6s,s,?p,p,5d,d', 2.9),
'Pt': ('6s,s,?p,p,5d,d', 3.0),
'Au': ('6s,s,?p,p,5d,d', 3.0),
'Hg': ('6s,s,?p,p,5d,d', 2.9),
'Tl': ('6s,s,6p,p,5d,d', 2.8),
'Pb': ('6s,s,6p,p,5d,d', 2.8),
'Bi': ('6s,s,6p,p,d', 3.0),
'Rn': ('6s,s,6p,p,d', 2.7)}


class PAWWaves:
    def __init__(self, rgd, l, rcut):
        self.rgd = rgd
        self.l = l
        self.rcut = rcut

        self.n_n = []
        self.e_n = []
        self.f_n = []
        self.phi_ng = []
        self.phit_ng = None
        self.pt_ng = None

    def __len__(self):
        return len(self.n_n)
    
    def add(self, phi_g, n, e, f):
        self.phi_ng.append(phi_g)
        self.n_n.append(n)
        self.e_n.append(e)
        self.f_n.append(f)

    def pseudize(self, old=False):
        rgd = self.rgd

        phi_ng = self.phi_ng = np.array(self.phi_ng)
        N = len(phi_ng)
        phit_ng = self.phit_ng = rgd.empty(N)
        gcut = rgd.ceil(self.rcut)

        P = 6
        if old:
            P = 4
        self.nt_g = 0
        self.c_np = []
        for n in range(N):
            phit_ng[n], c_p = rgd.pseudize(phi_ng[n], gcut, self.l, points=P)
            self.c_np.append(c_p)
            self.nt_g += self.f_n[n] / 4 / pi * phit_ng[n]**2
            
        self.dS_nn = np.empty((N, N))
        for n1 in range(N):
            for n2 in range(N):
                self.dS_nn[n1, n2] = rgd.integrate(
                    phi_ng[n1] * phi_ng[n2] -
                    phit_ng[n1] * phit_ng[n2]) / (4 * pi)
        self.Q = np.dot(self.f_n, self.dS_nn.diagonal())

    def construct_projectors(self, vtr_g):
        N = len(self)
        if N == 0:
            self.pt_ng = []
            return

        rgd = self.rgd
        phit_ng = self.phit_ng
        gcut = rgd.ceil(self.rcut)
        r_g = rgd.r_g
        l = self.l
        P = len(self.c_np[0]) - 1
        p = np.arange(2 * P, 0, -2) + l

        dgdr_g = 1 / rgd.dr_g
        d2gdr2_g = rgd.d2gdr2()

        q_ng = rgd.zeros(N)
        for n in range(N):
            a_g, dadg_g, d2adg2_g = rgd.zeros(3)
            a_g[1:] = self.phit_ng[n, 1:] / r_g[1:]**l
            a_g[0] = self.c_np[n][-1]
            dadg_g[1:-1] = 0.5 * (a_g[2:] - a_g[:-2])
            d2adg2_g[1:-1] = a_g[2:] - 2 * a_g[1:-1] + a_g[:-2]
            q_g = (vtr_g - self.e_n[n] * r_g) * self.phit_ng[n]
            q_g -= 0.5 * r_g**l * (
                (2 * (l + 1) * dgdr_g + r_g * d2gdr2_g) * dadg_g +
                r_g * d2adg2_g * dgdr_g**2)
            q_g[gcut:] = 0
            q_ng[n] = q_g

        A_nn = rgd.integrate(phit_ng[:, None] * q_ng, -1) / (4 * pi)
        self.dH_nn = self.e_n * self.dS_nn - A_nn

        L_nn = np.eye(N)
        U_nn = A_nn.copy()

        if 0:#N - self.n_n.count(-1) == 1:
            assert self.n_n[0] != -1
            # We have a single bound-state projector.
            for n1 in range(N):
                for n2 in range(n1 + 1, N):
                    L_nn[n2, n1] = U_nn[n2, n1] / U_nn[n1, n1]
                    U_nn[n2] -= U_nn[n1] * L_nn[n2, n1]

            iL_nn = np.linalg.inv(L_nn)
            phit_ng[:] = np.dot(iL_nn, phit_ng)
            self.phi_ng[:] = np.dot(iL_nn, self.phi_ng)

            self.dS_nn = np.dot(np.dot(iL_nn, self.dS_nn), iL_nn.T)
            self.dH_nn = np.dot(np.dot(iL_nn, self.dH_nn), iL_nn.T)

        self.pt_ng = np.dot(np.linalg.inv(U_nn.T), q_ng)
        self.pt_ng[:, 1:] /= r_g[1:]
        if l == 0:
            self.pt_ng[:, 0] = self.pt_ng[:, 1]

    def calculate_kinetic_energy_correction(self, vr_g, vtr_g):
        if len(self) == 0:
            return
        self.dekin_nn = (self.rgd.integrate(self.phit_ng[:, None] *
                                            self.phit_ng *
                                            vtr_g, -1) / (4 * pi) -
                         self.rgd.integrate(self.phi_ng[:, None] *
                                            self.phi_ng *
                                            vr_g, -1) / (4 * pi) +
                         self.dH_nn)
        print self.dekin_nn


class PAWSetupGenerator:
    def __init__(self, aea, projectors, rc, fd=sys.stdout, old=False):
        """fd: stream
            Text output."""
        
        self.aea = aea
        
        self.old = old

        if fd is None:
            fd = devnull
        self.fd = fd

        lmax = -1
        states = {}
        for s in projectors.split(','):
            l = 'spdf'.find(s[-1])
            if len(s) == 1:
                n = None
            elif '.' in s:
                n = float(s[:-1])
            else:
                n = int(s[:-1])
            if l in states:
                states[l].append(n)
            else:
                states[l] = [n]
            if l > lmax:
                lmax = l

        # Add empty bound states:
        for l, nn in states.items():
            for n in nn:
                if (isinstance(n, int) and
                    (l not in aea.f_lsn or n - l > len(aea.f_lsn[l][0]))):
                    aea.add(n, l, 0)

        for l in range(lmax):
            if l not in states:
                states[l] = []

        aea.initialize()
        aea.run()
        aea.refine()
        
        self.rgd = aea.rgd

        self.log('\nGenerating PAW setup for', aea.symbol)

        if isinstance(rc, float):
            radii = [rc]
        else:
            radii = rc

        self.rcmax = max(radii)
        self.gcmax = self.rgd.ceil(self.rcmax)

        if lmax >= 0:
            radii += [radii[-1]] * (lmax + 1 - len(radii))

        self.waves_l = []
        for l in range(lmax + 1):
            waves = PAWWaves(self.rgd, l, radii[l])
            e = -1.0
            for n in states[l]:
                if isinstance(n, int):
                    # Bound state:
                    ch = aea.channels[l]
                    e = ch.e_n[n - l - 1]
                    f = ch.f_n[n - l - 1]
                    phi_g = ch.phi_ng[n - l - 1]
                else:
                    if n is None:
                        e += 1.0
                    else:
                        e = n
                    n = -1
                    f = 0.0
                    phi_g = self.rgd.zeros()
                    gc = self.gcmax + 20
                    ch = Channel(l)
                    ch.integrate_outwards(phi_g, self.rgd, aea.vr_sg[0], gc, e)
                    phi_g[1:gc + 1] /= self.rgd.r_g[1:gc + 1]
                    if l == 0:
                        phi_g[0] = phi_g[1]
                    phi_g /= (self.rgd.integrate(phi_g**2) / (4*pi))**0.5

                waves.add(phi_g, n, e, f)
            self.waves_l.append(waves)

        self.construct_shape_function(eps=1e-9)

        self.vtr_g = None

    def construct_shape_function(self, eps):
        """Build shape-function for compensation charge."""

        def spillage(alpha):
            """Fraction of gaussian charge outside rcmax."""
            x = alpha * self.rcmax**2
            return 1 - erf(sqrt(x)) + 2 * sqrt(x / pi) * exp(-x)
        
        def f(alpha):
            return log(spillage(alpha)) - log(eps)

        self.alpha = fsolve(f, 7.0)[0]
        self.alpha = round(self.alpha, 1)
        if self.old:
            self.alpha = 10.0 / self.rcmax**2
        self.log('Shape function: exp(-alpha*r^2), alpha=%.1f Bohr^-2' %
                 self.alpha)
        self.log('Fraction of shape-function outside rcmax=%.2f Bohr: %e' %
                 (self.rcmax, spillage(self.alpha)))

        self.ghat_g = (np.exp(-self.alpha * self.rgd.r_g**2) *
                       (self.alpha / pi)**1.5)
        
    def log(self, *args, **kwargs):
        prnt(file=self.fd, *args, **kwargs)

    def calculate_core_density(self):
        self.nc_g = self.rgd.zeros()
        self.ncore = 0
        self.nvalence = 0
        self.ekincore = 0.0
        for l, ch in enumerate(self.aea.channels):
            for n, f in enumerate(ch.f_n):
                if (l >= len(self.waves_l) or
                    (l < len(self.waves_l) and
                     n + l + 1 not in self.waves_l[l].n_n)):
                    self.nc_g += f * ch.calculate_density(n)
                    self.ncore += f
                    self.ekincore += f * ch.e_n[n]
                else:
                    self.nvalence += f
        
        self.ekincore -= self.rgd.integrate(self.nc_g * self.aea.vr_sg[0], -1)
        self.nct_g = self.rgd.pseudize(self.nc_g, self.gcmax)[0]
        self.npseudocore = self.rgd.integrate(self.nct_g)

        self.log('Core electrons:', self.ncore)
        self.log('Pseudo core electrons: %.6f' %
                 self.rgd.integrate(self.nct_g))
        self.log('Valence electrons:', self.nvalence)
        
        self.nt_g = self.nct_g.copy()
        self.Q = -self.aea.Z + self.ncore - self.npseudocore

    def pseudize(self):
        for waves in self.waves_l:
            waves.pseudize(self.old)
            self.nt_g += waves.nt_g
            self.Q += waves.Q

        self.rhot_g = self.nt_g + self.Q * self.ghat_g
        self.vHtr_g = self.rgd.poisson(self.rhot_g)

        self.vxct_g = self.rgd.zeros()
        exct_g = self.rgd.zeros()
        self.exct = self.aea.xc.calculate_spherical(
            self.rgd, self.nt_g.reshape((1, -1)), self.vxct_g.reshape((1, -1)))

        self.log('\nProjectors:')
        self.log(' state  occ         energy             norm        rcut')
        self.log(' nl            [Hartree]  [eV]      [electrons]   [Bohr]')
        self.log('----------------------------------------------------------')
        for l, waves in enumerate(self.waves_l):
            for n, e, f, ds in zip(waves.n_n, waves.e_n, waves.f_n,
                                  waves.dS_nn.diagonal()):
                if n == -1:
                    self.log('  %s         %10.6f %10.5f   %19.2f' %
                             ('spdf'[l], e, e * Hartree, waves.rcut))
                else:
                    self.log(
                        ' %d%s     %2d  %10.6f %10.5f      %5.3f  %9.2f' %
                             (n, 'spdf'[l], f, e, e * Hartree, 1 - ds,
                              waves.rcut))
        self.log()
                    
    def find_polynomial_potential(self, r0, P, e0=None):
        g0 = self.rgd.ceil(r0)
        assert e0 is None
        if self.old:
            vtr_g = self.vHtr_g + self.vxct_g * self.rgd.r_g
            self.vtr_g = self.rgd.pseudize(vtr_g, g0, 1, P)[0]
            self.v0r_g = self.vtr_g - vtr_g
        else:
            self.vtr_g = self.rgd.pseudize(self.aea.vr_sg[0], g0, 1, P)[0]
            self.v0r_g = self.vtr_g - self.vHtr_g - self.vxct_g * self.rgd.r_g

        self.l0 = None
        self.e0 = None

    def find_local_potential(self, l0, r0, P, e0):
        self.log('Local potential matching %s-scattering at e=%.3f eV' %
                 ('spdfg'[l0], e0 * Hartree) +
                 ' and r=%.2f Bohr' % r0)
        
        g0 = self.rgd.ceil(r0)
        gc = g0 + 20

        ch = Channel(l0)
        phi_g = self.rgd.zeros()
        ch.integrate_outwards(phi_g, self.rgd, self.aea.vr_sg[0], gc, e0)
        phi_g[1:gc] /= self.rgd.r_g[1:gc]
        if l0 == 0:
            phi_g[0] = phi_g[1]

        phit_g, c_p = self.rgd.pseudize_normalized(phi_g, g0, l=l0, points=P)
        r_g = self.rgd.r_g[1:g0]
        p = np.arange(2 * P, 0, -2) + l0
        t_g = np.polyval(-0.5 * c_p[:P] * (p * (p + 1) - l0 * (l0 + 1)),
                          r_g**2) 

        dgdr_g = 1 / self.rgd.dr_g
        d2gdr2_g = self.rgd.d2gdr2()
        a_g = phit_g.copy()
        a_g[1:] /= self.rgd.r_g[1:]**l0
        a_g[0] = c_p[-1]
        dadg_g = self.rgd.zeros()
        d2adg2_g = self.rgd.zeros()
        dadg_g[1:-1] = 0.5 * (a_g[2:] - a_g[:-2])
        d2adg2_g[1:-1] = a_g[2:] - 2 * a_g[1:-1] + a_g[:-2]
        q_g = (((l0 + 1) * dgdr_g + 0.5 * self.rgd.r_g * d2gdr2_g) * dadg_g +
               0.5 * self.rgd.r_g * d2adg2_g * dgdr_g**2)
        q_g[:g0] /= a_g[:g0]
        q_g += e0 * self.rgd.r_g
        q_g[0] = 0.0

        self.vtr_g = self.aea.vr_sg[0].copy()
        self.vtr_g[0] = 0.0
        self.vtr_g[1:g0] = q_g[1:g0]#e0 * r_g - t_g * r_g**(l0 + 1) / phit_g[1:g0]
        self.v0r_g = self.vtr_g - self.vHtr_g - self.vxct_g * self.rgd.r_g
        #self.rgd.plot(q_g)
        #self.rgd.plot(self.vtr_g,show=1)
        #sdfg

        self.l0 = l0
        self.e0 = e0

    def construct_projectors(self):
        for waves in self.waves_l:
            waves.construct_projectors(self.vtr_g)
            waves.calculate_kinetic_energy_correction(self.aea.vr_sg[0],
                                                      self.vtr_g)

    def check(self):
        self.log('Checking eigenvalues of pseudo atom using ' +
                 'a Gaussian basis set:')
        self.log('           AE [Hartree]   PS [Hartree]   error [Hartree]')
        basis = self.aea.channels[0].basis
        eps = basis.eps
        alpha_B = basis.alpha_B

        ok = True

        for l in range(4):
            basis = GaussianBasis(l, alpha_B, self.rgd, eps)
            H_bb = basis.calculate_potential_matrix(self.vtr_g)
            H_bb += basis.T_bb
            S_bb = np.eye(len(basis))
            
            n0 = 0
            if l < len(self.waves_l):
                waves = self.waves_l[l]
                if len(waves) > 0:
                    P_bn = self.rgd.integrate(basis.basis_bg[:, None] *
                                              waves.pt_ng) / (4 * pi)
                    H_bb += np.dot(np.dot(P_bn, waves.dH_nn), P_bn.T)
                    S_bb += np.dot(np.dot(P_bn, waves.dS_nn), P_bn.T)
                    n0 = waves.n_n[0] - l - 1
                    if n0 < 0 and l < len(self.aea.channels):
                        n0 = (self.aea.channels[l].f_n > 0).sum()
            elif l < len(self.aea.channels):
                n0 = (self.aea.channels[l].f_n > 0).sum()                

            e_b = np.empty(len(basis))
            general_diagonalize(H_bb, e_b, S_bb)
            nbound = (e_b < 0).sum()

            if l < len(self.aea.channels):
                e0_b = self.aea.channels[l].e_n
                nbound0 = (e0_b < 0).sum()
                extra = 6
                for n in range(1 + l, nbound0 + 1 + l + extra):
                    if n - 1 - l < len(self.aea.channels[l].f_n):
                        f = self.aea.channels[l].f_n[n - 1 - l]
                        self.log('%2d%s  %2d' % (n, 'spdf'[l], f), end='')
                    else:
                        self.log('       ', end='')
                    self.log('  %15.6f' % e0_b[n - 1 - l], end='')
                    if n - 1 - l - n0 >= 0:
                        self.log('%15.6f' * 2 %
                                 (e_b[n - 1 - l - n0],
                                  e_b[n - 1 - l - n0] - e0_b[n - 1 - l]))
                    else:
                        self.log()
                
                if nbound != nbound0 - n0:
                    self.log('Wrong number of %s-states!' % 'spdf'[l])
                    ok = False
                elif (nbound > 0 and
                      abs(e_b[:nbound] - e0_b[n0:nbound0]).max() > 1e-3):
                    self.log('Error in bound %s-states!' % 'spdf'[l])
                    ok = False
                elif (abs(e_b[nbound:nbound + extra] -
                          e0_b[nbound0:nbound0 + extra]).max() > 2e-2):
                    self.log('Error in %s-states!' % 'spdf'[l])
                    ok = False
            elif nbound > 0:
                self.log('Wrong number of %s-states!' % 'spdf'[l])
                ok = False

        return ok

    def plot(self):
        import matplotlib.pyplot as plt
        r_g = self.rgd.r_g

        plt.figure()
        plt.plot(r_g, self.vxct_g, label='xc')
        plt.plot(r_g[1:], self.v0r_g[1:] / r_g[1:], label='0')
        plt.plot(r_g[1:], self.vHtr_g[1:] / r_g[1:], label='H')
        plt.plot(r_g[1:], self.vtr_g[1:] / r_g[1:], label='ps')
        plt.plot(r_g[1:], self.aea.vr_sg[0, 1:] / r_g[1:], label='ae')
        plt.axis(xmax=2 * self.rcmax,
                 ymin=self.vtr_g[1] / r_g[1],
                 ymax=max(0, (self.v0r_g[1:] / r_g[1:]).max()))
        plt.legend()
        
        plt.figure()
        i = 0
        for l, waves in enumerate(self.waves_l):
            for n, e, phi_g, phit_g in zip(waves.n_n, waves.e_n,
                                           waves.phi_ng, waves.phit_ng):
                if n == -1:
                    gc = self.rgd.ceil(waves.rcut)
                    name = '*%s (%.2f Ha)' % ('spdf'[l], e)
                else:
                    gc = len(self.rgd)
                    name = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
                plt.plot(r_g[:gc], (phi_g * r_g)[:gc], color=colors[i],
                         label=name)
                plt.plot(r_g[:gc], (phit_g * r_g)[:gc], '--', color=colors[i])
                i += 1
        plt.axis(xmax=3 * self.rcmax)
        plt.legend()

        plt.figure()
        i = 0
        for l, waves in enumerate(self.waves_l):
            for n, e, pt_g in zip(waves.n_n, waves.e_n, waves.pt_ng):
                if n == -1:
                    name = '*%s (%.2f Ha)' % ('spdf'[l], e)
                else:
                    name = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
                plt.plot(r_g, pt_g * r_g, color=colors[i], label=name)
                i += 1
        plt.axis(xmax=self.rcmax)
        plt.legend()

    def logarithmic_derivative(self, l, energies, rcut):
        rgd = self.rgd
        ch = Channel(l)
        gcut = rgd.round(rcut)

        N = 0
        if l < len(self.waves_l):
            # Nonlocal PAW stuff:
            waves = self.waves_l[l]
            if len(waves) > 0:
                pt_ng = waves.pt_ng
                dH_nn = waves.dH_nn
                dS_nn = waves.dS_nn
                N = len(pt_ng)

        u_g = rgd.zeros()
        u_ng = rgd.zeros(N)
        dudr_n = np.empty(N)

        logderivs = []
        for e in energies:
            dudr = ch.integrate_outwards(u_g, rgd, self.vtr_g, gcut, e)
            u = u_g[gcut]
            
            if N:
                for n in range(N):
                    dudr_n[n] = ch.integrate_outwards(u_ng[n], rgd,
                                                      self.vtr_g, gcut, e,
                                                      pt_ng[n])
            
                A_nn = (dH_nn - e * dS_nn) / (4 * pi)
                B_nn = rgd.integrate(pt_ng[:, None] * u_ng, -1)
                c_n  = rgd.integrate(pt_ng * u_g, -1)
                d_n = np.linalg.solve(np.dot(A_nn, B_nn) + np.eye(N),
                                      np.dot(A_nn, c_n))
                u -= np.dot(u_ng[:, gcut], d_n)
                dudr -= np.dot(dudr_n, d_n)
            
            logderivs.append(dudr / u)

        return logderivs
    
    def make_paw_setup(self):
        aea = self.aea
        
        setup = SetupData(aea.symbol, aea.xc.name, 'paw', readxml=False)

        nj = sum(len(waves) for waves in self.waves_l)
        setup.e_kin_jj = np.zeros((nj, nj))
        setup.id_j = []
        j1 = 0
        for l, waves in enumerate(self.waves_l):
            ne = 0
            for n, f, e, phi_g, phit_g, pt_g in zip(waves.n_n, waves.f_n,
                                                    waves.e_n, waves.phi_ng,
                                                    waves.phit_ng,
                                                    waves.pt_ng):
                setup.append(n, l, f, e, waves.rcut, phi_g, phit_g, pt_g)
                if n == -1:
                    ne += 1
                    id = '%s-%s%d' % (aea.symbol, 'spdf'[l], ne)
                else:
                    id = '%s-%d%s' % (aea.symbol, n, 'spdf'[l])
                setup.id_j.append(id)
            j2 = j1 + len(waves)
            setup.e_kin_jj[j1:j2, j1:j2] = waves.dekin_nn
            j1 = j2

        setup.nc_g = self.nc_g * sqrt(4 * pi)
        setup.nct_g = self.nct_g * sqrt(4 * pi)
        setup.e_kinetic_core = self.ekincore
        setup.vbar_g = self.v0r_g * sqrt(4 * pi)
        setup.vbar_g[1:] /= self.rgd.r_g[1:]
        setup.vbar_g[0] = setup.vbar_g[1]
        setup.Z = aea.Z
        setup.Nc = self.ncore
        setup.Nv = self.nvalence
        setup.e_kinetic = aea.ekin
        setup.e_xc = aea.exc
        setup.e_electrostatic = aea.eH + aea.eZ
        setup.e_total = aea.exc + aea.ekin + aea.eH + aea.eZ
        setup.rgd = self.rgd
        setup.rcgauss = 1 / sqrt(self.alpha)

        setup.tauc_g = self.rgd.zeros()
        setup.tauct_g = self.rgd.zeros()
        print 'no tau!!!!!!!!!!!'
        
        reltype = 'non-relativistic'
        attrs = [('type', reltype), ('name', 'gpaw-%s' % version)]
        setup.generatorattrs = attrs

        return setup

     
def build_parser(): 
    from optparse import OptionParser

    parser = OptionParser(usage='%prog [options] element',
                          version='%prog 0.1')
    parser.add_option('-f', '--xc-functional', type='string', default='LDA',
                      help='Exchange-Correlation functional ' +
                      '(default value LDA)',
                      metavar='<XC>')
    parser.add_option('-P', '--projectors',
                      help='Projector functions - use comma-separated - ' +
                      'nl values, where n can be pricipal quantum number ' +
                      '(integer) or energy (floating point number). ' +
                      'Example: 2s,0.5s,2p,0.5p,0.0d.')
    parser.add_option('-r', '--radius',
                      help='1.2 or 1.2,1.1,1.1')
    parser.add_option('-0', '--zero-potential',
                      metavar='type,nderivs,radius,e0',
                      help='Parameters for zero potential.')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-l', '--logarithmic-derivatives',
                      metavar='spdfg,e1:e2:de,radius',
                      help='Plot logarithmic derivatives. ' +
                      'Example: -l spdf,-1:1:0.05,1.3. ' +
                      'Energy range and/or radius can be left out.')
    parser.add_option('-w', '--write', action='store_true')
    parser.add_option('--old', action='store_true')
    return parser


def main(AEA=AllElectronAtom):
    parser = build_parser()
    opt, args = parser.parse_args()

    if len(args) != 1:
        parser.error('Incorrect number of arguments')
    symbol = args[0]

    kwargs = {'xc': opt.xc_functional}
        
    aea = AEA(symbol, **kwargs)

    projectors, radii = parameters[symbol][:2]

    if opt.projectors:
        projectors = opt.projectors
                
    if opt.radius:
        radii = [float(r) for r in opt.radius.split(',')]

    gen = PAWSetupGenerator(aea, projectors, radii, old=opt.old)

    gen.calculate_core_density()
    gen.pseudize()

    if opt.zero_potential:
        x = opt.zero_potential.split(',')
        type = x[0]
        nderiv = int(x[1])
        r0 = float(x[2])
        if len(x) == 4:
            e0 = float(x[3])
        elif type == 'poly':
            e0 = None
        else:
            e0 = 0.0

        if type == 'poly':
            gen.find_polynomial_potential(r0, nderiv, e0)
        else:
            l0 = 'spdfg'.find(type)
            gen.find_local_potential(l0, r0, nderiv, e0)
    else:
        if len(parameters[symbol]) == 2:
            gen.find_polynomial_potential(gen.rcmax, 6)
        else:
            gen.find_local_potential(3, gen.rcmax, 6, 0.0)
        
    gen.construct_projectors()

    ok = gen.check()

    if ok and opt.write:
        gen.make_paw_setup().write_xml()
        
    if opt.logarithmic_derivatives or opt.plot:
        import matplotlib.pyplot as plt
        if opt.logarithmic_derivatives:
            r = 1.1 * gen.rcmax
            emin = min(min(wave.e_n) for wave in gen.waves_l) - 0.2
            emax = max(max(wave.e_n) for wave in gen.waves_l) + 0.2
            lvalues, energies, r = parse_ld_str(opt.logarithmic_derivatives,
                                                (emin, emax, 0.05), r)
            ldmax = 0.0
            for l in lvalues:
                ld = aea.logarithmic_derivative(l, energies, r)
                plt.plot(energies, ld, colors[l], label='spdfg'[l])
                ld = gen.logarithmic_derivative(l, energies, r)
                plt.plot(energies, ld, '--' + colors[l])

                # Fixed points:
                if l < len(gen.waves_l):
                    efix = gen.waves_l[l].e_n
                    ldfix = gen.logarithmic_derivative(l, efix, r)
                    plt.plot(efix, ldfix, 'x' + colors[l])
                    ldmax = max(ldmax, max(abs(ld) for ld in ldfix))

                if l == gen.l0:
                    efix = [gen.e0]
                    ldfix = gen.logarithmic_derivative(l, efix, r)
                    plt.plot(efix, ldfix, 'x' + colors[l])
                    ldmax = max(ldmax, max(abs(ld) for ld in ldfix))


            if ldmax != 0.0:
                plt.axis(ymin=-3 * ldmax, ymax=3 * ldmax)
            plt.legend(loc='best')
            

        if opt.plot:
            gen.plot()

        plt.show()


if __name__ == '__main__':
    main()

class PAWSetup:
    def __init__(self, alpha, r_g, phit_g, v0_g):
        self.natoms = 0
        self.E = 0.0
        self.Z = 1
        self.Nc = 0
        self.Nv = 1
        self.niAO = 1
        self.pt_j = []
        self.ni = 0
        self.l_j = []
        self.nct = None
        self.Nct = 0.0

        rc = 1.0
        r2_g = np.linspace(0, rc, 100)**2
        x_g = np.exp(-alpha * r2_g)
        x_g[-1] = 0 

        self.ghat_l = [Spline(0, rc,
                              (4 * pi)**0.5 * (alpha / pi)**1.5 * x_g)]

        self.vbar = Spline(0, rc, (4 * pi)**0.5 * v0_g[0] * x_g)

        r = np.linspace(0, 4.0, 100)
        phit = splev(r, splrep(r_g, phit_g))
        poly = np.polyfit(r[[-30,-29,-2,-1]], [0, 0, phit[-2], phit[-1]], 3)
        phit[-30:] -= np.polyval(poly, r[-30:])
        self.phit_j = [Spline(0, 4.0, phit)]
                              
        self.Delta_pL = np.zeros((0, 1))
        self.Delta0 = -1 / (4 * pi)**0.5
        self.lmax = 0
        self.K_p = self.M_p = self.MB_p = np.zeros(0)
        self.M_pp = np.zeros((0, 0))
        self.Kc = 0.0
        self.MB = 0.0
        self.M = 0.0
        self.xc_correction = null_xc_correction
        self.HubU = None
        self.dO_ii = np.zeros((0, 0))
        self.type = 'local'
        self.fingerprint = None
        
    def get_basis_description(self):
        return '1s basis cut off at 4 Bohr'

    def print_info(self, text):
        text('Local pseudo potential')
        
    def calculate_initial_occupation_numbers(self, magmom, hund, charge,
                                             nspins):
        return np.array([(1.0,)])

    def initialize_density_matrix(self, f_si):
        return np.zeros((1, 0))

    def calculate_rotations(self, R_slmm):
        self.R_sii = np.zeros((1, 0, 0))

if 0:
    for Z in range(1, 44):
        aea = AllElectronAtom(Z)
        symbol = chemical_symbols[Z]
        p = parameters[symbol]
        projectors, radii = p[:2]
        gen = PAWSetupGenerator(aea, projectors, radii)
        gen.calculate_core_density()
        gen.pseudize()
        if len(p) == 2:
            gen.find_polynomial_potential(gen.rcmax, 6)
        else:
            gen.find_local_potential(3, gen.rcmax, 6, 0.0)
        gen.construct_projectors()
        assert gen.check()
        #gen.make_paw_setup().write_xml()
 
