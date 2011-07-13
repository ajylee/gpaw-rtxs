import numpy as np
import copy
import sys
from math import pi, sqrt
from time import time, ctime
from datetime import timedelta
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw import GPAW
from sigma import SIGMA
from gpaw.response.parallel import parallel_partition
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.response.df import DF
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.utilities import devnull
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc

class GW:

    def __init__(
                 self,
                 file=None,
                 bands=None,
                 kpoints=None,
                 w=None,
                 ecut=150.,
                 eta=0.1,
                 txt=None,
                ):

        self.file = file
        self.bands = bands
        self.kpoints = kpoints
        self.w_w = w
        self.ecut = ecut
        self.eta = eta
        self.txtname = txt

        self.output_init()


    def get_QP_spectrum(self):

        calc = GPAW(self.file, communicator=serial_comm, txt=None)
        kd = calc.wfs.kd
        nkpt = kd.nbzkpts
        nikpt = kd.nibzkpts
        self.bzk_kc = kd.bzk_kc
        self.ibzk_kc = kd.ibzk_kc
        nbands = calc.wfs.nbands

        starttime = time()

        self.printtxt("------------------------------------------------")
        self.printtxt('starting calculation at %s' %(ctime(starttime)))

        # k-points init
        self.printtxt("------------------------------------------------")
        self.printtxt('calculate matrix elements for k = :')

        if (self.kpoints == None):
            nkptout = nikpt
            kptout = range(nikpt)
            for k in kptout:
                self.printtxt(kd.ibzk_kc[k])
        else:
            nkptout = np.shape(self.kpoints)[0]
            kptout = self.kpoints
            for k in kptout:
                self.printtxt(kd.bzk_kc[k])

        # bands init
        if (self.bands == None):
            nbandsout = nbands
            bandsout = range(nbands)
        else:
            nbandsout = np.shape(self.bands)[0]
            bandsout = self.bands
        self.printtxt("------------------------------------------------")
        self.printtxt('calculate matrix elements for n = :')
        self.printtxt(bandsout)

        frequencies = self.w_w
        energy_cut = self.ecut
        broadening = self.eta

        wmin = frequencies.min()
        wmax = frequencies.max()
        Nw = np.shape(frequencies)[0] - 1
        wstep = (wmax - wmin) / Nw

        self.printtxt("------------------------------------------------")
        self.printtxt("broadening (eV):")
        self.printtxt(broadening)
        self.printtxt("plane wave cut-off (eV):")
        self.printtxt(energy_cut)
        self.printtxt("frequency range (eV):")
        self.printtxt('%.2f - %.2f in %.2f' %(wmin, wmax, wstep))
        self.printtxt("------------------------------------------------")
        self.printtxt("number of bands:")
        self.printtxt(nbands)
        self.printtxt("number of k-points:")
        self.printtxt(nkpt)
        self.printtxt("number of irreducible k-points:")
        self.printtxt(nikpt)
        self.printtxt("------------------------------------------------")

        bzq_kc = kd.get_bz_q_points()
        nqpt = np.shape(bzq_kc)[0]

        nG = calc.get_number_of_grid_points()
        acell_cv = calc.atoms.cell / Bohr
        acell_cv, bcell_cv, vol, BZvol = get_primitive_cell(acell_cv, np.array([1,1,1]))
        npw, Gvec_Gc, Gindex_G = set_Gvectors(acell_cv, bcell_cv, nG, np.ones(3) * energy_cut / Hartree)

        Sigma_kn = np.zeros((nkptout, nbandsout), dtype=float)
        Z_kn = np.zeros((nkptout, nbandsout), dtype=float)

        qcomm = world
        nq, nq_local, q_start, q_end = parallel_partition(nqpt, world.rank, world.size, reshape=False)

        self.printtxt("calculating Sigma")
        self.printtxt("------------------------------------------------")

        for iq in range(q_start, q_end):
            q = bzq_kc[iq]

            w = copy.copy(frequencies)
            ecut = copy.copy(energy_cut)
            eta = copy.copy(broadening)

            q_G = np.zeros(npw, dtype=float)
            dfinv_wGG = np.zeros((Nw, npw, npw), dtype = complex)
            W_wGG = np.zeros((Nw, npw, npw), dtype = complex)
            tmp_GG = np.eye(npw, npw)

            self.printtxt('%i calculating q = [%f %f %f]' %(iq, q[0], q[1], q[2]))
            self.printtxt("------------------------------------------------")

            if (np.abs(q) < 1e-5).all():
                q0 = np.array([0.0001, 0., 0.])
                optical_limit = True
            else:
                q0 = q
                optical_limit = False

            qG = np.dot(q0[np.newaxis,:] + Gvec_Gc,(bcell_cv).T)
            q_G = 1. / np.sqrt((qG*qG).sum(axis=1))

            if optical_limit:
                q0 = np.array([1e-10, 0., 0.])

            df = DF(
                    calc=calc,
                    q=q0,
                    w=copy.copy(w),
                    nbands=nbands,
                    optical_limit=False,
                    hilbert_trans=True,
                    full_response=True,
                    xc='RPA',
                    eta=eta,
                    ecut=ecut,
                    txt='df_q_' + str(iq) + '.out',
                    comm=serial_comm
                   )

            dfinv_wGG = df.get_inverse_dielectric_matrix(xc='RPA')

            for iw in range(Nw):
                W_wGG[iw] =  4*pi * ((q_G[:,np.newaxis] * (dfinv_wGG[iw] - tmp_GG)) * q_G[np.newaxis,:])

            del df
            del q_G, dfinv_wGG

            sigma = SIGMA(
                          calc=calc,
                          nbands=nbands,
                          bands=bandsout,
                          kpoints=kptout,
                          w=w,
                          q=q,
                          ecut=ecut,
                          eta=eta,
                          txt='gw_q_' + str(iq) + '.out',
                          optical_limit=optical_limit
                         )

            S, Z = sigma.get_self_energy(W_wGG)

            Sigma_kn += S
            Z_kn += Z

            del q0, q, w, eta, ecut

        qcomm.barrier()
        qcomm.sum(Sigma_kn)
        qcomm.sum(Z_kn)

        Z_kn /= nkpt
        Sigma_kn /= nkpt

        self.printtxt("calculating V_XC")
        self.printtxt("------------------------------------------------")
        v_xc = vxc(calc)

#        calc.set(parallel={'domain': 1})

        self.printtxt("calculating E_XX")
        alpha = 5.0
        exx = HybridXC('EXX', alpha=alpha)
        calc.get_xc_difference(exx)

        e_kn = np.zeros((nkptout, nbandsout), dtype=float)
        v_kn = np.zeros((nkptout, nbandsout), dtype=float)
        e_xx = np.zeros((nkptout, nbandsout), dtype=float)

        i = 0
        for k in kptout:
            j = 0
            ik = kd.bz2ibz_k[k]
            for n in bandsout:
                e_kn[i][j] = calc.get_eigenvalues(kpt=ik)[n] / Hartree
                v_kn[i][j] = v_xc[0][ik][n] / Hartree
                e_xx[i][j] = exx.exx_skn[0][ik][n]
                j += 1
            i += 1

        self.printtxt("------------------------------------------------")
        self.printtxt("LDA eigenvalues are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(e_kn*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("LDA exchange-correlation contributions are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(v_kn*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("exact exchange contributions are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(e_xx*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("correlation contributions are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(Sigma_kn*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("renormalization factors are:")
        self.printtxt("------------------------------------------------")
        self.printtxt(Z_kn)

        Sigma_kn = e_kn + Z_kn * (Sigma_kn + e_xx - v_kn)

        totaltime = round(time() - starttime)

        self.printtxt("------------------------------------------------")
        self.printtxt("GW calculation finished!")
        self.printtxt('in %s' %(timedelta(seconds=totaltime)))
        self.printtxt("------------------------------------------------")
        self.printtxt("Quasi-particle energies are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(Sigma_kn*Hartree)


    def output_init(self):

        if self.txtname is None:
            if rank == 0:
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull
        elif self.txtname == devnull:
            self.txt = devnull
        else:
            assert type(self.txtname) is str
            self.txt = paropen(self.txtname,'w')


    def printtxt(self, text):
        print >> self.txt, text
