import numpy as np

import _gpaw
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
from gpaw.xc.vdw import FFTVDWFunctional
from gpaw import debug


class BEE1(XCKernel):
    """GGA exchange expanded in a PBE-like basis."""
    def __init__(self, parameters=None):
        """BEE1.

        parameters : array
            [thetas,coefs] for the basis expansion.

        """

        if parameters is None:
            self.name = 'BEE1'
            parameters = [0.0, 1.0]
        else:
            self.name = 'BEE1?'
        parameters = np.array(parameters, dtype=float).ravel()
        self.xc = _gpaw.XCFunctional(18, parameters)
        self.type = 'GGA'


class BEE2(XCKernel):
    """GGA exchange expanded in Legendre polynomials."""
    def __init__(self, parameters=None):
        """BEE2.

        parameters: array
            [transformation,0.0,[orders],[coefs]].

        """

        if parameters is None:
            # LDA exchange
            t = [1.0, 0.0]
            coefs = [1.0]
            orders = [0.0]
            parameters = np.append(t, np.append(orders, coefs))
        else:
            assert len(parameters) > 2
            assert np.mod(len(parameters), 2) == 0
            assert parameters[1] == 0.0

        parameters = np.array(parameters, dtype=float).ravel()
        self.xc = _gpaw.XCFunctional(17, parameters)
        self.type = 'GGA'
        self.name = 'BEE2'


class BEEVDWKernel(XCKernel):
    """Kernel for BEEVDW functionals."""
    def __init__(self, bee, xcoefs, ldac, pbec):
        """BEEVDW kernel.

        parameters:

        bee : str
            choose BEE1 or BEE2 exchange basis expansion.
        xcoefs : array
            coefficients for exchange.
        ldac : float
            coefficient for LDA correlation.
        pbec : float
            coefficient for PBE correlation.

        """

        if bee is 'BEE1':
            self.BEE = BEE1(xcoefs)
        elif bee is 'BEE2':
            self.BEE = BEE2(xcoefs)
        else:
            raise ValueError('Unknown BEE exchange: %s', bee)

        self.LDAc = LibXC('LDA_C_PW')
        self.PBEc = LibXC('GGA_C_PBE')
        self.ldac = ldac
        self.pbec = pbec

        self.type = 'GGA'
        self.name = 'BEEVDW'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)

        self.BEE.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)

        e0_g = np.empty_like(e_g)
        dedn0_sg = np.empty_like(dedn_sg)
        dedsigma0_xg = np.empty_like(dedsigma_xg)
        for coef, kernel in [
            (self.ldac, self.LDAc),
            (self.pbec - 1.0, self.PBEc)]:
            dedn0_sg[:] = 0.0
            kernel.calculate(e0_g, n_sg, dedn0_sg, sigma_xg, dedsigma0_xg)
            e_g += coef * e0_g
            dedn_sg += coef * dedn0_sg
            if kernel.type == 'GGA':
                dedsigma_xg += coef * dedsigma0_xg


class BEEVDWFunctional(FFTVDWFunctional):
    """Base class for BEEVDW functionals."""
    def __init__(self, bee='BEE1', xcoefs=(0.0, 1.0),
                 ccoefs=(0.0, 1.0, 0.0), t=4.0, orders=None,
                 **kwargs):
        """BEEVDW functionals.

        parameters:

        bee : str
            choose BEE1 or BEE2 exchange basis expansion.
        xcoefs : array-like
            coefficients for exchange.
        ccoefs : array-like
            LDA, PBE, nonlocal correlation coefficients
        t : float
            transformation for BEE2 exchange
        orders : array
            orders of Legendre polynomials for BEE2 exchange

        """

        if bee is 'BEE1':
            name = 'BEE1VDW'
            Zab = -0.8491
            soft_corr = False
        elif bee is 'BEE2':
            name = 'BEE2VDW'
            Zab = -1.887
            soft_corr = False
            if orders is None:
                orders = range(len(xcoefs))
            xcoefs = np.append([t, 0.0], np.append(orders, xcoefs))
        elif bee == 'BEEF-vdW':
            bee = 'BEE2'
            name = 'BEEF-vdW'
            Zab = -1.887
            soft_corr = True
            t, x, o, ccoefs = self.load_xc_pars('BEEF-vdW')
            xcoefs = np.append(t, np.append(o, x))
            self.t, self.x, self.o, self.c = t, x, o, ccoefs
            self.nl_type = 2
        else:
            raise KeyError('Unknown BEEVDW functional: %s', bee)

        ldac, pbec, vdw = ccoefs
        kernel = BEEVDWKernel(bee, xcoefs, ldac, pbec)
        FFTVDWFunctional.__init__(self, name=name, soft_correction=soft_corr,
                                  kernel=kernel, Zab=Zab, vdwcoef=vdw,
                                  **kwargs)

    def get_setup_name(self):
        return 'PBE'

    def load_xc_pars(self, name):
        if name == 'BEEF-vdW':
            from beefvdw_pars import t, x, o, c
            return t, x, o, c
        else:
            raise KeyError('Unknown XC name: %s', name)


class BEEF_Ensemble:
    """BEEF ensemble error estimation."""
    def __init__(self, calc=None, exch=None, corr=None):
        """BEEF ensemble

        parameters:

        calc : object
            Calculator holding a selfconsistent BEEF electron density.
        exch : array
            Exchange basis function contributions to the total energy.
            Defaults to None.
        corr : array
            Correlation basis function contributions to the total energy.
            Defaults to None.

        """

        self.calc = calc
        self.exch = exch
        self.corr = corr
        if self.calc is None:
            raise KeyError('calculator not specified')

        # determine functional and read parameters
        self.xc = self.calc.get_xc_functional()
        if self.xc in ['BEEF-vdW', 'BEEF-1']:
            self.bee = BEEVDWFunctional('BEEF-vdW')
            self.nl_type = self.bee.nl_type
            self.t = self.bee.t
            self.x = self.bee.x
            self.o = self.bee.o
            self.c = self.bee.c
        else:
            raise NotImplementedError('xc = %s not implemented' % self.xc)

    def get_ensemble_energies(self, ensemble_size=2000, seed=0):
        """Returns an array of ensemble total energies"""

        if self.exch is None:
            x = self.beef_energy_contribs_x()
        else:
            x = self.exch
        if self.corr is None:
            c = self.beef_energy_contribs_c()
        else:
            c = self.corr
        assert len(x) == 30
        assert len(c) == 2

        basis_constribs = np.append(x, c)
        ensemble_coefs = self.get_ensemble_coefs(ensemble_size, seed)
        de = np.dot(ensemble_coefs, basis_constribs)
        return de

    def get_ensemble_coefs(self, ensemble_size, seed):
        """Pertubation coefficients of BEEF ensemble functionals."""

        if self.xc in ['BEEF-vdW', 'BEEF-1']:
            from beefvdw_pars import uiOmega

            N = ensemble_size
            assert np.shape(uiOmega) == (31, 31)
            Wo, Vo = np.linalg.eig(uiOmega)
            np.random.seed(seed)
            RandV = np.random.randn(31, N)

            for j in range(N):
                v = RandV[:,j]
                coefs_i = (np.dot(np.dot(Vo, np.diag(np.sqrt(Wo))), v)[:])
                if j == 0:
                    ensemble_coefs = coefs_i
                else:
                    ensemble_coefs = np.vstack((ensemble_coefs, coefs_i))
            PBEc_ens = -ensemble_coefs[:, 30]
            ensemble_coefs = (np.vstack((ensemble_coefs.T, PBEc_ens))).T
        else:
            raise NotImplementedError('xc = %s not implemented' % self.xc)
        return ensemble_coefs

    def beef_energy_contribs_x(self):
        """Legendre polynomial exchange contributions to Etot"""
        from gpaw.xc import XC
        from gpaw.xc.kernel import XCNull

        e_dft = self.calc.get_potential_energy()
        xc_null = XC(XCNull())
        e_0 = e_dft + self.calc.get_xc_difference(xc_null)
        e_pbe = e_dft + self.calc.get_xc_difference('GGA_C_PBE') - e_0

        exch = np.zeros(len(self.o))
        for p in self.o:
            pars = [self.t[0], self.t[1], p, 1.0]
            bee = XC('BEE2', pars)
            exch[p] = e_dft + self.calc.get_xc_difference(bee) - e_0 - e_pbe
            del bee
        return exch

    def beef_energy_contribs_c(self):
        """LDA and PBE correlation contributions to Etot"""
        from gpaw.xc import XC
        from gpaw.xc.kernel import XCNull

        e_dft = self.calc.get_potential_energy()
        xc_null = XC(XCNull())
        e_0 = e_dft + self.calc.get_xc_difference(xc_null)
        e_lda = e_dft + self.calc.get_xc_difference('LDA_C_PW') - e_0
        e_pbe = e_dft + self.calc.get_xc_difference('GGA_C_PBE') - e_0
        corr = np.array([e_lda, e_pbe])
        return corr
