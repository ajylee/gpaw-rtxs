#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, exp, sqrt, log

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy import __version__ as scipy_version
from ase.utils import prnt
from ase.units import Hartree, Bohr
from ase.data import atomic_numbers, chemical_symbols

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
'H':  ('1s,s,p', 1.0, {}),
'He': ('1s,s,p', 1.3, {}),
'Li': ('2s,2.0s,2p', 2.6, {}),
'Be': ('2s,s,2p', 2.0, {}),
'B':  ('2s,s,2p,p,d', 1.4, {'gamma': 1.0, 'h': 0.0}),
'C':  ('2s,s,2p,p,d', 1.3, {'gamma': 1.8}),
'N':  ('2s,s,2p,p,d', 1.3, {'gamma': 1.8}),
'O':  ('2s,s,2p,p,d', 1.5, {}),
'F':  ('2s,s,2p,p,d', 1.4, {'gamma': 1.7}),
'Ne': ('2s,s,2p,p,d', 1.8, {}),
'Na': ('3s,s,3p,p,d', 3.0, {'local': 'f'}),
'Mg': ('3s,s,3p,p,d', 2.8, {}),
'Al': ('3s,s,3p,p,d', 2.2, {}),
'Si': ('3s,s,3p,p,d', 2.4, {}),
'P':  ('3s,s,3p,p,d', 1.9, {}),
'S':  ('3s,s,3p,p,d', 2.2, {}),
'Cl': ('3s,s,3p,p,d', 2.1, {}),
'Ar': ('3s,s,3p,p,d', 2.2, {}),
'K':  ('3s,4s,3p,4p,d', 2.4, {}),
'Ca': ('3s,4s,3p,4p,3d', 2.5, {}),
'Sc': ('3s,4s,3p,4p,3d,d', 2.7, {'local': 'f'}),
'Ti': ('3s,4s,3p,4p,3d,d', 2.6, {'local': 'f'}),
'V':  ('3s,4s,3p,4p,3d,d', 2.5, {'local': 'f'}),
'Cr': ('3s,4s,3p,4p,3d,d', 2.3, {'local': 'f'}),
'Mn': ('4s,s,4p,p,3d,d', 2.8, {}),
'Fe': ('4s,s,4p,p,3d,d', 2.8, {}),
'Co': ('4s,s,4p,p,3d,d', 2.7, {}),
'Ni': ('4s,s,4p,p,3d,d', 2.7, {}),
'Cu': ('4s,s,4p,p,3d,d', 2.5, {}),
'Zn': ('4s,s,4p,p,3d,d', 2.4, {}),
'Ga': ('4s,s,4p,p,3d,d', 2.4, {}),
'Ge': ('4s,s,4p,p,d', 2.7, {}),
'As': ('4s,s,4p,p,d', 2.7, {}),
'Se': ('4s,5s,4p,p,d', 2.7, {}),
'Br': ('4s,5s,4p,p,d', 2.6, {}),
'Kr': ('4s,5s,4p,p,d', 2.4, {}),
'Rb': ('4s,5s,4p,5p,d', 2.7, {'local': 'f'}),
'Sr': ('4s,5s,4p,5p,4d,d', 2.5, {'local': 'f'}),
'Y':  ('4s,5s,4p,5p,4d,d', 2.6, {'local': 'f'}),
'Zr': ('4s,5s,4p,5p,4d,d', 2.7, {'local': 'f'}),
'Nb': ('4s,5s,4p,5p,4d,d', 2.8, {'local': 'f'}),
'Mo': ('4s,5s,4p,5p,4d,d', 2.7, {'local': 'f'}),
'Tc': ('4s,5s,4p,5p,4d,d', 2.6, {'local': 'f'}),
'Ru': ('4s,5s,4p,5p,4d,d', 2.5, {'local': 'f'}),
'Rh': ('5s,s,5p,p,4d,d', 3.2, {}),
'Pd': ('5s,s,5p,p,4d,d', 3.1, {}),
'Ag': ('5s,s,5p,p,4d,d', 3.1, {}),
'Cd': ('5s,s,5p,p,4d,d', 3.0, {}),
'In': ('5s,s,5p,p,4d,d', 2.9, {}),
'Sn': ('5s,s,5p,p,d', 2.8, {}),
'Sb': ('5s,s,5p,p,d', 2.8, {}),
'Te': ('5s,s,5p,p,d', 2.7, {}),
'I':  ('5s,s,5p,p,d', 2.7, {}),
'Xe': ('5s,s,5p,p,d', 2.8, {}),
'Cs': ('5s,6s,5p,p,d', 3.2, {}),
'Ba': ('5s,6s,5p,p,d', [2.5, 3.0], {}),
'La': ('5s,6s,s,5p,6p,5d,d', 3.0, {}),
'Ce': ('5s,6s,s,5p,6p,p,5d,d', [3.3, 2.8], {}),
'Hf': ('5s,6s,5p,6p,5d,d', 3.1, {}),
'Ta': ('5s,6s,5p,6p,5d,d', 3.0, {}),
'W':  ('5s,6s,5p,6p,5d,d', 3.0, {}),
'Re': ('5s,6s,5p,6p,5d,d', 3.0, {}),
'Os': ('6s,s,6p,p,5d,d', 3.5, {}),
'Ir': ('6s,s,6p,p,5d,d', 3.4, {}),
'Pt': ('6s,s,6p,p,5d,d', 3.3, {}),
'Au': ('6s,s,6p,p,5d,d', 3.2, {}),
'Hg': ('6s,s,6p,p,5d,d', 3.2, {}),
'Tl': ('6s,s,6p,p,5d,d', 3.2, {}),
'Pb': ('6s,s,6p,p,5d,d', 3.1, {}),
'Bi': ('6s,s,6p,p,d', 3.0, {})
}

extra_parameters = {
'Ru': ('4s,5s,s,4p,5p,p,4d,d', 2.9, {'local': 'f'}),
'Ru': ('5s,s,5p,p,4d,d', [3.3, 3.3, 2.9], {}),
}

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

    def pseudize(self):
        rgd = self.rgd

        phi_ng = self.phi_ng = np.array(self.phi_ng)
        N = len(phi_ng)
        phit_ng = self.phit_ng = rgd.empty(N)
        gcut = rgd.ceil(self.rcut)

        P = 6
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

    def construct_projectors(self, vtr_g, rcmax, rcfilter, Gcut):
        N = len(self)
        if N == 0:
            self.pt_ng = []
            return

        rgd = self.rgd
        phit_ng = self.phit_ng
        gcmax = rgd.ceil(rcmax)
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
            q_g[gcmax:] = 0
            q_g[1:] /= r_g[1:]
            if l == 0:
                q_g[0] = q_g[1]
            if Gcut is not None:
                q_g = rgd.filter(q_g, rcfilter, Gcut, l)
            q_ng[n] = q_g

        A_nn = rgd.integrate(phit_ng[:, None] * q_ng) / (4 * pi)
        self.dH_nn = self.e_n * self.dS_nn - A_nn

        L_nn = np.eye(N)
        U_nn = A_nn.copy()
        
        if N - self.n_n.count(-1) == 1:
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


class PAWSetupGenerator:
    def __init__(self, aea, projectors, rc,
                 scalar_relativistic=False, alpha=None,
                 gamma=1.5, h=0.2 / Bohr, fd=sys.stdout):
        """fd: stream
            Text output."""
        
        self.aea = aea

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
        aea.scalar_relativistic = scalar_relativistic
        aea.refine()
        
        self.rgd = aea.rgd

        self.log('\nGenerating PAW', aea.xc.name, 'setup for', aea.symbol)

        if isinstance(rc, float):
            radii = [rc]
        else:
            radii = rc

        self.rcmax = max(radii)
        self.gcmax = self.rgd.ceil(self.rcmax)

        self.rcfilter = self.rcmax * gamma
        if h == 0:
            self.Gcut = None
        else:
            self.Gcut = pi / h - 2 / self.rcfilter
            self.log('Wang mask function for Fourier filtering:')
            self.log('gamma=%.2f, h=%.2f Bohr, rcut=%.2f, Gcut=%.2f Bohr^-1' %
                     (gamma, h, self.rcfilter, self.Gcut))

        if lmax >= 0:
            radii += [radii[-1]] * (lmax + 1 - len(radii))

        self.waves_l = []
        for l in range(lmax + 1):
            rcut = radii[l]
            waves = PAWWaves(self.rgd, l, rcut)
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
                    gc = self.rgd.round(self.rcfilter) + 10
                    ch = Channel(l)
                    ch.integrate_outwards(phi_g, self.rgd, aea.vr_sg[0], gc, e,
                                          aea.scalar_relativistic)
                    phi_g[1:gc + 1] /= self.rgd.r_g[1:gc + 1]
                    if l == 0:
                        phi_g[0] = phi_g[1]
                    phi_g /= (self.rgd.integrate(phi_g**2) / (4*pi))**0.5

                waves.add(phi_g, n, e, f)
            self.waves_l.append(waves)

        self.alpha = alpha
        self.construct_shape_function(eps=1e-10)

        self.vtr_g = None

    def construct_shape_function(self, eps):
        """Build shape-function for compensation charge."""

        if self.alpha is None:
            rc = 1.5 * self.rcmax
            def spillage(alpha):
                """Fraction of gaussian charge outside rc."""
                x = alpha * rc**2
                return 1 - erf(sqrt(x)) + 2 * sqrt(x / pi) * exp(-x)
        
            def f(alpha):
                return log(spillage(alpha)) - log(eps)

            if scipy_version < '0.8':
                self.alpha = fsolve(f, 7.0)
            else:
                self.alpha = fsolve(f, 7.0)[0]

            self.alpha = round(self.alpha, 1)

        self.log('Shape function: exp(-alpha*r^2), alpha=%.1f Bohr^-2' %
                 self.alpha)

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

        self.log('Core electrons:', self.ncore)
        self.log('Valence electrons:', self.nvalence)
        
        self.Q = -self.aea.Z + self.ncore

    def pseudize(self):
        self.nt_g = self.rgd.zeros()
        for waves in self.waves_l:
            waves.pseudize()
            self.nt_g += waves.nt_g
            self.Q += waves.Q

        self.nct_g = self.rgd.pseudize(self.nc_g, self.gcmax)[0]
        self.nt_g += self.nct_g

        # Make sure pseudo density is monotonically decreasing:
        dntdr_g = self.rgd.derivative(self.nt_g)[:self.gcmax]
        if dntdr_g.max() > 0.0:
            # Constuct function that decrease smoothly from
            # f(0)=1 to f(rcmax)=0:
            x_g = self.rgd.r_g[:self.gcmax] / self.rcmax
            f_g = self.rgd.zeros()
            f_g[:self.gcmax] = (1 - x_g**2 * (3 - 2 * x_g))**2

            # Add enough of f to nct to make nt monotonically decreasing:
            dfdr_g = self.rgd.derivative(f_g)
            A = (-dntdr_g / dfdr_g[:self.gcmax]).max() * 1.5
            self.nt_g += A * f_g
            self.nct_g += A * f_g
            
        self.npseudocore = self.rgd.integrate(self.nct_g)
        self.log('Pseudo core electrons: %.6f' % self.npseudocore)
        self.Q -= self.npseudocore

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

        self.vtr_g = self.rgd.pseudize(self.aea.vr_sg[0], g0, 1, P)[0]
        self.v0r_g = self.vtr_g - self.vHtr_g - self.vxct_g * self.rgd.r_g
        self.v0r_g[g0:] = 0.0

        self.l0 = None
        self.e0 = None

        self.filter_zero_potential()

    def find_local_potential(self, l0, r0, P, e0):
        self.log('Local potential matching %s-scattering at e=%.3f eV' %
                 ('spdfg'[l0], e0 * Hartree) +
                 ' and r=%.2f Bohr' % r0)
        
        g0 = self.rgd.ceil(r0)
        gc = g0 + 20

        ch = Channel(l0)
        phi_g = self.rgd.zeros()
        ch.integrate_outwards(phi_g, self.rgd, self.aea.vr_sg[0], gc, e0,
                              self.aea.scalar_relativistic)
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
        self.v0r_g[g0:] = 0.0

        self.l0 = l0
        self.e0 = e0

        self.filter_zero_potential()

    def filter_zero_potential(self):
        if self.Gcut is not None:
            self.vtr_g -= self.v0r_g
            self.v0r_g[1:] /= self.rgd.r_g[1:]
            self.v0r_g[0] = self.v0r_g[1]
            self.v0r_g = self.rgd.filter(
                self.v0r_g, self.rcfilter, self.Gcut) * self.rgd.r_g
            self.vtr_g += self.v0r_g

    def construct_projectors(self):
        for waves in self.waves_l:
            waves.construct_projectors(self.vtr_g, self.rcmax,
                                       self.rcfilter, self.Gcut)
            waves.calculate_kinetic_energy_correction(self.aea.vr_sg[0],
                                                      self.vtr_g)

    def check(self):
        self.log(('Checking eigenvalues of %s pseudo atom using ' +
                  'a Gaussian basis set:') % self.aea.symbol)
        self.log('                 AE [eV]        PS [eV]      error [eV]')
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
            try:
                general_diagonalize(H_bb, e_b, S_bb)
            except RuntimeError:
                self.log('Singular overlap matrix!')
                ok = False
                continue
            
            nbound = (e_b < -0.002).sum()

            if l < len(self.aea.channels):
                e0_b = self.aea.channels[l].e_n
                nbound0 = (e0_b < -0.002).sum()
                extra = 6
                for n in range(1 + l, nbound0 + 1 + l + extra):
                    if n - 1 - l < len(self.aea.channels[l].f_n):
                        f = self.aea.channels[l].f_n[n - 1 - l]
                        self.log('%2d%s  %2d' % (n, 'spdf'[l], f), end='')
                    else:
                        self.log('       ', end='')
                    self.log('  %15.3f' % (e0_b[n - 1 - l] * Hartree), end='')
                    if n - 1 - l - n0 >= 0:
                        self.log('%15.3f' * 2 %
                                 (e_b[n - 1 - l - n0] * Hartree,
                                  (e_b[n - 1 - l - n0] - e0_b[n - 1 - l]) *
                                  Hartree))
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

    def test_convergence(self):
        rgd = self.rgd
        r_g = rgd.r_g
        G_k, nt_k = self.rgd.fft(self.nt_g * r_g)
        rhot_k = self.rgd.fft(self.rhot_g * r_g)[1]
        ghat_k = self.rgd.fft(self.ghat_g * r_g)[1]
        v0_k = self.rgd.fft(self.v0r_g)[1]
        vt_k = self.rgd.fft(self.vtr_g)[1]
        phi_k = self.rgd.fft(self.waves_l[0].phit_ng[0] * r_g)[1]
        eee_k = 0.5 * nt_k**2 * (4 * pi)**2 / (2 * pi)**3
        ecc_k = 0.5 * rhot_k**2 * (4 * pi)**2 / (2 * pi)**3
        egg_k = 0.5 * ghat_k**2 * (4 * pi)**2 / (2 * pi)**3
        ekin_k = 0.5 * phi_k**2 * G_k**4 / (2 * pi)**3
        evt_k = nt_k * vt_k * G_k**2 * 4 * pi / (2 * pi)**3

        eee = 0.5 * rgd.integrate(self.nt_g * rgd.poisson(self.nt_g), -1)
        ecc = 0.5 * rgd.integrate(self.rhot_g * self.vHtr_g, -1)
        egg = 0.5 * rgd.integrate(self.ghat_g * rgd.poisson(self.ghat_g), -1)
        ekin = self.aea.ekin - self.ekincore - self.waves_l[0].dekin_nn[0, 0] 
        print self.aea.ekin, self.ekincore, self.waves_l[0].dekin_nn[0, 0] 
        evt = rgd.integrate(self.nt_g * self.vtr_g, -1)

        import pylab as p

        errors = 10.0**np.arange(-4, 0) / Hartree
        self.log('\nConvergence of energy:')
        self.log('plane-wave cutoff (wave-length) [ev (Bohr)]\n  ', end='')
        for de in errors:
            self.log('%14.4f' % (de * Hartree), end='')
        for label, e_k, e0 in [
            ('e-e', eee_k, eee),
            ('c-c', ecc_k, ecc),
            ('g-g', egg_k, egg),
            ('kin', ekin_k, ekin),
            ('vt', evt_k, evt)]:
            self.log('\n%3s: ' % label, end='')
            e_k = (np.add.accumulate(e_k) - 0.5 * e_k[0] - 0.5 * e_k) * G_k[1]
            print e_k[-1],e0, e_k[-1]-e0
            k = len(e_k) - 1
            for de in errors:
                while abs(e_k[k] - e_k[-1]) < de:
                    k -= 1
                G = k * G_k[1]
                ecut = 0.5 * G**2
                h = pi / G
                self.log(' %6.1f (%4.2f)' % (ecut * Hartree, h), end='')
            p.semilogy(G_k, abs(e_k - e_k[-1]) * Hartree, label=label)
        self.log()
        p.axis(xmax=20)
        p.xlabel('G')
        p.ylabel('[eV]')
        p.legend()
        p.show()

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
        plt.xlabel('radius [Bohr]')
        plt.ylabel('potential [Ha]')
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
        plt.xlabel('radius [Bohr]')
        plt.ylabel(r'$r\phi_{n\ell}(r)$')
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
                                                      pt_g=pt_ng[n])
            
                A_nn = (dH_nn - e * dS_nn) / (4 * pi)
                B_nn = rgd.integrate(pt_ng[:, None] * u_ng, -1)
                c_n  = rgd.integrate(pt_ng * u_g, -1)
                d_n = np.linalg.solve(np.dot(A_nn, B_nn) + np.eye(N),
                                      np.dot(A_nn, c_n))
                u -= np.dot(u_ng[:, gcut], d_n)
                dudr -= np.dot(dudr_n, d_n)
            
            logderivs.append(dudr / u)

        return logderivs
    
    def make_paw_setup(self, tag=None):
        aea = self.aea
        
        setup = SetupData(aea.symbol, aea.xc.name, tag, readxml=False)

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
        
        if self.aea.scalar_relativistic:
            reltype = 'scalar-relativistic'
        else:
            reltype = 'non-relativistic'
        attrs = [('type', reltype), ('name', 'gpaw-%s' % version)]
        setup.generatorattrs = attrs

        return setup


def str2z(x):
    if isinstance(x, int):
        return x
    if x[0].isdigit():
        return int(x)
    return atomic_numbers[x]


def generate(argv=None):
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
    parser.add_option('-s', '--scalar-relativistic', action='store_true')
    parser.add_option('--no-check', action='store_true')
    parser.add_option('-t', '--tag', type='string')
    parser.add_option('-c', '--convergence', action='store_true')
    parser.add_option('-a', '--alpha', type=float)
    parser.add_option('-F', '--filter', metavar='gamma,h',
                      help='Fourrier filtering parameters for Wang ' +
                      'mask-function.  Default: ' +
                      u'gamma=1.5 and h=0.2 Å.  Use gamma=1 and ' +
                      u'h=0 Å to turn off filtering.')

    opt, args = parser.parse_args(argv)

    if len(args) == 0:
        symbols = range(1, 87)
    elif len(args) == 1 and '-' in args[0]:
        Z1, Z2 = args[0].split('-')
        Z1 = str2z(Z1)
        if Z2:
            Z2 = str2z(Z2)
        else:
            Z2 = 86
        symbols = range(Z1, Z2 + 1)
    else: 
        symbols = args
                    
    for symbol in symbols:
        Z = str2z(symbol)
        symbol = chemical_symbols[Z]

        gen = _generate(symbol, opt)

    return gen


def _generate(symbol, opt):
    aea = AllElectronAtom(symbol, xc=opt.xc_functional)

    if symbol in parameters:
        projectors, radii, extra = parameters[symbol]
    else:
        projectors, radii, extra = None, 1.0, {}
    
    if opt.projectors:
        projectors = opt.projectors
                
    if opt.radius:
        radii = [float(r) for r in opt.radius.split(',')]

    gamma = extra.get('gamma', 1.5)
    h = extra.get('h', 0.2)  # Angstrom

    if opt.filter:
        gamma, h = (float(x) for x in opt.filter.split(','))

    gen = PAWSetupGenerator(aea, projectors, radii,
                            opt.scalar_relativistic,
                            opt.alpha, gamma, h / Bohr)

    gen.calculate_core_density()
    gen.pseudize()

    if opt.zero_potential:
        x = opt.zero_potential.split(',')
        type = x[0]
        if len(x) == 1:
            # select only zero_potential type (with defaults)
            # i.e. on the command line: -0 {f,poly}
            nderiv = 6
            r0 = gen.rcmax
        else:
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
        if 'local' not in extra:
            gen.find_polynomial_potential(gen.rcmax, 6)
        else:
            l0 = 'spdfg'.find(extra['local'])
            gen.find_local_potential(l0, gen.rcmax, 6, 0.0)
        
    gen.construct_projectors()

    if opt.no_check:
        ok = True
    else:
        ok = gen.check()

    if opt.convergence:
        gen.test_convergence()
        
    if opt.write:
        gen.make_paw_setup(opt.tag).write_xml()
        
    if opt.logarithmic_derivatives or opt.plot:
        import matplotlib.pyplot as plt
        if opt.logarithmic_derivatives:
            r = 1.1 * gen.rcmax
            emin = min(min(wave.e_n) for wave in gen.waves_l) - 0.8
            emax = max(max(wave.e_n) for wave in gen.waves_l) + 0.8
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
            plt.xlabel('energy [Ha]')
            plt.ylabel(r'$d\phi_{\ell\epsilon}(r)/dr/\phi_{\ell\epsilon}(r)|_{r=r_c}$')
            plt.legend(loc='best')
            

        if opt.plot:
            gen.plot()

        plt.show()
    
    return gen


if __name__ == '__main__':
    generate()
