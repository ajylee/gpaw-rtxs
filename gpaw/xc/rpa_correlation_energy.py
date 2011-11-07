import sys
from time import ctime
import numpy as np
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.response.df import DF
from gpaw.utilities import devnull
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import rank, size, world
from gpaw.response.parallel import parallel_partition, set_communicator

class RPACorrelation:

    def __init__(self, calc, txt=None, qsym=True, xc=None):
        
        self.calc = calc
        
        if txt is None:
            if rank == 0:
                #self.txt = devnull
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull
        else:
            assert type(txt) is str
            from ase.parallel import paropen
            self.txt = paropen(txt, 'w')

        self.qsym = qsym
        self.nspins = calc.wfs.nspins
        self.bz_k_points = calc.wfs.bzk_kc
        self.atoms = calc.get_atoms()
        self.setups = calc.wfs.setups
        self.bz_q_points = calc.wfs.kd.get_bz_q_points()
        if qsym == False:
            self.ibz_q_points = self.bz_q_points
            self.q_weights = (np.ones(len(self.bz_q_points))
                              / len(self.bz_q_points))
        else:
            op_scc = calc.wfs.kd.symmetry.op_scc
            self.ibz_q_points = calc.wfs.kd.get_ibz_q_points(self.bz_q_points,
                                                             op_scc)[0]
            self.q_weights = calc.wfs.kd.q_weights
        
        if xc == None:
            self.xc = 'RPA'
        else:
            self.xc = xc

        self.print_initialization()
        self.initialized = 0
   
    def get_rpa_correlation_energy(self,
                                   kcommsize=1,
                                   dfcommsize=world.size,
                                   directions=None,
                                   skip_gamma=False,
                                   ecut=10,
                                   nbands=None,
                                   gauss_legendre=None,
                                   frequency_cut=None,
                                   frequency_scale=None,
                                   w=None,
                                   extrapolate=False,
                                   restart=None):
            
        self.initialize_calculation(w,
                                    ecut,
                                    nbands,
                                    kcommsize,
                                    extrapolate,
                                    gauss_legendre,
                                    frequency_cut,
                                    frequency_scale)

        if dfcommsize == world.size:
            self.dfcomm = world
            E_q = []
            if restart is not None:
                assert type(restart) is str
                try:
                    f = paropen(restart, 'r')
                    lines = f.readlines()
                    for line in lines:
                        E_q.append(eval(line))
                    f.close()
                    print >> self.txt, 'Correlation energy from %s q-points obtained from restart file: ' % len(E_q), restart
                    print >> self.txt
                except:
                    IOError
    
            for index, q in enumerate(self.ibz_q_points[len(E_q):]):
                if abs(q[0]) < 0.001 and abs(q[1]) < 0.001 and abs(q[2]) < 0.001:
                    E_q0 = 0.
                    if skip_gamma:
                        print >> self.txt, 'Not calculating q at the Gamma point'
                        print >> self.txt
                    else:
                        if directions is None:
                            directions = [[0, 1/3.], [1, 1/3.], [2, 1/3.]]
                        for d in directions:                                   
                            E_q0 += self.E_q(q, index=index, direction=d[0]) * d[1]
                    E_q.append(E_q0)
                else:
                    E_q.append(self.E_q(q, index=index))
                    
                if restart is not None:
                    f = paropen(restart, 'a')
                    print >> f, E_q[-1]
                    f.close()
    
            E = np.dot(np.array(self.q_weights), np.array(E_q).real)

        else: # parallelzation over q points
            print >> self.txt, 'parallelization over q point ! '
            # creast q list
            qlist = []
            qweight = []
            id = 0
            for iq, q in enumerate(self.ibz_q_points):
                if abs(q[0]) < 0.001 and abs(q[1]) < 0.001 and abs(q[2]) < 0.001:
                    if skip_gamma:
                        continue
                    else:
                        if directions is None:
                            directions = [[0, 1/3.], [1, 1/3.], [2, 1/3.]]
                        for d in directions:
                            qlist.append((id, q, d[0], d[1]))
                            qweight.append(self.q_weights[iq])
                            id += 1
                        continue
                qlist.append((id, q, 0, 1))
                qweight.append(self.q_weights[iq])
                id += 1
            nq = len(qlist)
    
            # distribute q list
            self.dfcomm, qcomm = set_communicator(world, world.rank, world.size, kcommsize=dfcommsize)[:2]
            nq, nq_local, q_start, q_end = parallel_partition(
                                      nq, qcomm.rank, qcomm.size, reshape=False)
    
            E_q = np.zeros(nq)
            for iq in range(q_start, q_end):
                E_q[iq] = self.E_q(qlist[iq][1], index=iq, direction=qlist[iq][2]) * qlist[iq][3]
            qcomm.sum(E_q)
    
            print >> self.txt, '(q, direction, weight), E_q, qweight'
            for iq in range(nq):
                print >> self.txt, qlist[iq][1:4], E_q[iq], qweight[iq]
    
            E = np.dot(np.array(qweight), np.array(E_q))

        print >> self.txt, '%s correlation energy:' % self.xc
        print >> self.txt, 'E_c = %s eV' % E
        print >> self.txt
        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt
        return E

    def get_E_q(self,
                kcommsize=1,
                index=None,
                q=[0., 0., 0.],
                direction=0,
                integrated=True,
                ecut=10,
                nbands=None,
                gauss_legendre=None,
                frequency_cut=None,
                frequency_scale=None,
                w=None,
                extrapolate=False):

        self.initialize_calculation(w, ecut, nbands, kcommsize, extrapolate,
                                    gauss_legendre, frequency_cut,
                                    frequency_scale)
        self.dfcomm = world
        E_q = self.E_q(q, direction=direction, integrated=integrated)
        
        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, '------------------------------------------------------'

        return E_q


    def E_q(self,
            q,
            index=None,
            direction=0,
            integrated=True):
        
        if abs(q[0]) < 0.001 and abs(q[1]) < 0.001 and abs(q[2]) < 0.001:
            q = [0.,0.,0.]
            q[direction] = 1.e-5
            optical_limit = True
        else:
            optical_limit = False

        dummy = DF(calc=self.calc,
                   xc=self.xc,
                   eta=0.0,
                   q=q,
                   w=self.w * 1j,
                   ecut=self.ecut,
                   G_plus_q=True,
                   kcommsize=self.kcommsize,
                   comm=self.dfcomm,
                   optical_limit=optical_limit,
                   hilbert_trans=False)

        dummy.txt = devnull
        dummy.initialize(simple_version=True)
        npw = dummy.npw
        del dummy
        if self.nbands is None:
            nbands = npw
        else:
            nbands = self.nbands

        df = DF(calc=self.calc,
                xc=self.xc,
                nbands=nbands,
                eta=0.0,
                q=q,
                txt='no_output',
                w=self.w * 1j,
                ecut=self.ecut,
                G_plus_q=True,
                kcommsize=self.kcommsize,
                comm=self.dfcomm,
                optical_limit=optical_limit,
                hilbert_trans=False)
        #df.txt = devnull
        
        if index is None:
            print >> self.txt, 'Calculating KS response function at:'
        else:
            print >> self.txt, '#', index, '- Calculating KS response function at:'
        
        if optical_limit:
            print >> self.txt, 'q = [0 0 0] -', 'Polarization: ', direction
        else:
            print >> self.txt, 'q = [%1.4f %1.4f %1.4f] -' % (q[0],q[1],q[2]), '%s planewaves' % npw

        if self.xc == 'RPA':
            e_wGG = df.get_dielectric_matrix(xc=self.xc)
            Nw_local = len(e_wGG)
            local_E_q_w = np.zeros(Nw_local, dtype=complex)
            E_q_w = np.empty(len(self.w), complex)
            for i in range(Nw_local):
                local_E_q_w[i] = (np.log(np.linalg.det(e_wGG[i]))
                                  + len(e_wGG[0]) - np.trace(e_wGG[i]))
                #local_E_q_w[i] = (np.sum(np.log(np.linalg.eigvals(e_wGG[i])))
                #                  + len(e_wGG[0]) - np.trace(e_wGG[i]))
            df.wcomm.all_gather(local_E_q_w, E_q_w)
            del e_wGG
        else:
            df.initialize()
            df.calculate()
            Kc_GG = np.zeros((df.npw, df.npw), dtype=complex)
            for iG in range(df.npw):
                qG = np.dot(df.q_c + df.Gvec_Gc[iG], df.bcell_cv)
                Kc_GG[iG,iG] = 4 * np.pi / np.dot(qG, qG)
            fhxc_GG = Kc_GG + df.Kxc_GG
            Nw_local = len(df.chi0_wGG)
            local_E_q_w = np.zeros(Nw_local, dtype=complex)
            E_q_w = np.empty(len(self.w), complex)
            for i in range(Nw_local):
                eigenvalues, P = np.linalg.eig(np.eye(df.npw, df.npw)
                                               - np.dot(df.chi0_wGG[i], fhxc_GG))
                chi0_v = np.dot(df.chi0_wGG[i], Kc_GG)
                chi0_fhxc = np.dot(df.chi0_wGG[i], fhxc_GG)
                A = np.dot(chi0_v, np.linalg.inv(chi0_fhxc))
                B = np.dot(np.linalg.inv(P), A)
                C = np.dot(B, P)
                local_E_q_w[i] = (np.dot(np.diag(C), np.log(eigenvalues))
                                  + np.trace(chi0_v))
            df.wcomm.all_gather(local_E_q_w, E_q_w)
        del df
            
        if self.gauss_legendre is not None:
            E_q = np.sum(E_q_w * self.gauss_weights * self.transform) / (4*np.pi)
        else:   
            dws = self.w[1:] - self.w[:-1]
            E_q = np.dot((E_q_w[:-1] + E_q_w[1:])/2., dws) / (2.*np.pi)

            if self.extrapolate:
                '''Fit tail to: Eq(w) = A**2/((w-B)**2 + C)**2'''
                e1 = abs(E_q_w[-1])**0.5
                e2 = abs(E_q_w[-2])**0.5
                e3 = abs(E_q_w[-3])**0.5
                w1 = self.w[-1]
                w2 = self.w[-2]
                w3 = self.w[-3]
                B = (((e3*w3**2-e1*w1**2)/(e1-e3) - (e2*w2**2-e1*w1**2)/(e1-e2))
                     / ((2*w3*e3-2*w1*e1)/(e1-e3) - (2*w2*e2-2*w1*e1)/(e1-e2)))
                C = ((w2-B)**2*e2 - (w1-B)**2*e1)/(e1-e2)
                A = e1*((w1-B)**2+C)
                if C > 0:
                    E_q -= A**2*(np.pi/(4*C**1.5)
                                 - (w1-B)/((w1-B)**2+C)/(2*C)
                                 - np.arctan((w1-B)/C**0.5)/(2*C**1.5)) / (2*np.pi)
                else:
                    E_q += A**2*((w1-B)/((w1-B)**2+C)/(2*C)
                                 + np.log((w1-B-abs(C)**0.5)/(w1-B+abs(C)**0.5))
                                 /(4*C*abs(C)**0.5)) / (2*np.pi)

        print >> self.txt, 'E_c(q) = %s eV' % E_q.real
        print >> self.txt

        if integrated:
            return E_q.real
        else:
            return E_q_w.real       

    def print_initialization(self):
        
        print >> self.txt, '------------------------------------------------------'
        if self.xc == 'RPA':
            print >> self.txt, 'Non-self-consistent RPA correlation energy'
        else:
            print >> self.txt, 'Non-self-consistent %s correlation energy' % self.xc
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt, 'Started at:  ', ctime()
        print >> self.txt
        print >> self.txt, 'Atoms                          :   %s' % self.atoms.get_name()
        print >> self.txt, 'Ground state XC functional     :   %s' % self.calc.hamiltonian.xc.name
        print >> self.txt, 'Valence electrons              :   %s' % self.setups.nvalence
        print >> self.txt, 'Number of Bands                :   %s' % self.calc.wfs.nbands
        print >> self.txt, 'Number of Converged Bands      :   %s' % self.calc.input_parameters['convergence']['bands']
        print >> self.txt, 'Number of Spins                :   %s' % self.nspins
        print >> self.txt, 'Number of k-points             :   %s' % len(self.calc.wfs.bzk_kc)
        print >> self.txt, 'Number of q-points             :   %s' % len(self.bz_q_points)
        print >> self.txt, 'Number of Irreducible k-points :   %s' % len(self.calc.wfs.ibzk_kc)
        if self.qsym:
            print >> self.txt, 'Number of Irreducible q-points :   %s' % len(self.ibz_q_points)
        else:
            print >> self.txt, 'No reduction of q-points' 
        print >> self.txt
        for q, weight in zip(self.ibz_q_points, self.q_weights):
            print >> self.txt, 'q: [%1.3f %1.3f %1.3f] - weight: %1.3f' % (q[0],q[1],q[2],
                                                                           weight)
        print >> self.txt
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt
        

    def initialize_calculation(self, w, ecut, nbands, kcommsize, extrapolate,
                               gauss_legendre, frequency_cut, frequency_scale):
        if w is not None:
            assert (gauss_legendre is None and
                    frequency_cut is None and
                    frequency_scale is None)
        else:
            if gauss_legendre is None:
                gauss_legendre = 16
            self.gauss_points, self.gauss_weights = self.get_weights_and_abscissas(N=gauss_legendre)
            if frequency_scale is None:
                frequency_scale = 2.0
            if frequency_cut is None:
                frequency_cut = 800.
            ys = 0.5 - 0.5 * self.gauss_points
            ys = ys[::-1]
            w = (-np.log(1-ys))**frequency_scale
            w *= frequency_cut/w[-1]
            alpha = (-np.log(1-ys[-1]))**frequency_scale/frequency_cut
            transform = (-np.log(1-ys))**(frequency_scale-1)/(1-ys)*frequency_scale/alpha
            self.transform = transform

        dummy = DF(calc=self.calc,
                   xc=self.xc,
                   eta=0.0,
                   w=w * 1j,
                   q=[0.,0.,0.0001],
                   ecut=ecut,
                   optical_limit=True,
                   hilbert_trans=False,
                   kcommsize=kcommsize)
        dummy.txt = devnull
        dummy.spin = 0
        dummy.initialize(simple_version=True)

        self.ecut = ecut
        self.w = w
        self.gauss_legendre = gauss_legendre
        self.frequency_cut = frequency_cut
        self.frequency_scale = frequency_scale
        self.extrapolate = extrapolate
        self.kcommsize = kcommsize
        self.nbands = nbands
        
        print >> self.txt, 'Planewave cut off             : %s eV' % ecut
        print >> self.txt, 'Number of Planewaves at Gamma : %s' % dummy.npw
        if self.nbands is None:
            print >> self.txt, 'Response function bands       : Equal to number of Planewaves'
        else:
            print >> self.txt, 'Response function bands       : %s' % self.nbands
        print >> self.txt, 'Frequencies'
        if self.gauss_legendre is not None:
            print >> self.txt, '    Gauss-Legendre integration with %s frequency points' % len(self.w)
            print >> self.txt, '    Frequency cutoff is %s eV and scale (B) is %s' % (self.w[-1], self.frequency_scale)
        else:
            print >> self.txt, '    %s specified frequency points' % len(self.w)
            print >> self.txt, '    Frequency cutoff is %s eV' % self.w[-1]
            if extrapolate:
                print >> self.txt, '    Squared Lorentzian extrapolation to frequencies at infinity'
        print >> self.txt
        print >> self.txt, 'Parallelization scheme'
        print >> self.txt, '     Total CPUs         : %d' % dummy.comm.size
        if dummy.nkpt == 1:
            print >> self.txt, '     Band parsize       : %d' % dummy.kcomm.size
        else:
            print >> self.txt, '     Kpoint parsize     : %d' % dummy.kcomm.size
        print >> self.txt, '     Frequency parsize  : %d' % dummy.wScomm.size
        print >> self.txt, 'Memory usage estimate'
        print >> self.txt, '     chi0_wGG(Q)        : %f M / cpu' % (dummy.Nw_local *
                                                                     dummy.npw**2 * 16.
                                                                    / 1024**2)
        print >> self.txt
        del dummy


    def get_weights_and_abscissas(self, N):
        #only works for N = 8, 16, 24 and 32
        if N == 8:
            weights = np.array([0.10122853629,
                                0.222381034453,
                                0.313706645878,
                                0.362683783378,
                                0.362683783378,
                                0.313706645878,
                                0.222381034453,
                                0.10122853629])
            abscissas = np.array([-0.960289856498,
                                  -0.796666477414,
                                  -0.525532409916,
                                  -0.183434642496,
                                  0.183434642496,
                                  0.525532409916,
                                  0.796666477414, 
                                  0.960289856498])    
        
        if N == 16:
            weights = np.array([0.027152459411,
                                0.0622535239372,
                                0.0951585116838,
                                0.124628971256,
                                0.149595988817,
                                0.169156519395,
                                0.182603415045,
                                0.189450610455,
                                0.189450610455,
                                0.182603415045,
                                0.169156519395,
                                0.149595988817,
                                0.124628971256,
                                0.0951585116838,
                                0.0622535239372,
                                0.027152459411])
            abscissas = np.array([-0.989400934992,
                                  -0.944575023073,    
                                  -0.865631202388, 	
                                  -0.755404408355, 	
                                  -0.617876244403, 	
                                  -0.458016777657, 	
                                  -0.281603550779, 	
                                  -0.0950125098376, 	
                                  0.0950125098376, 	
                                  0.281603550779,
                                  0.458016777657,
                                  0.617876244403,
                                  0.755404408355,
                                  0.865631202388,
                                  0.944575023073,
                                  0.989400934992])

        if N == 24:
            weights = np.array([ 0.01234123,
                                 0.02853139,
                                 0.04427744,
                                 0.05929859,
                                 0.07334648,
                                 0.08619016,
                                 0.09761865,
                                 0.10744427,
                                 0.11550567,
                                 0.12167047,
                                 0.12583746,
                                 0.1279382,
                                 0.1279382,
                                 0.12583746,
                                 0.12167047,
                                 0.11550567,
                                 0.10744427,
                                 0.09761865,
                                 0.08619016,
                                 0.07334648,
                                 0.05929859,
                                 0.04427744,
                                 0.02853139,
                                 0.01234123])
            
            abscissas = np.array([-0.99518722,
                                  -0.97472856,
                                  -0.93827455,
                                  -0.88641553,
                                  -0.82000199,
                                  -0.74012419,
                                  -0.64809365,
                                  -0.54542147,
                                  -0.43379351,
                                  -0.31504268,
                                  -0.19111887,
                                  -0.06405689,
                                  0.06405689,
                                  0.19111887,
                                  0.31504268,
                                  0.43379351,
                                  0.54542147,
                                  0.64809365,
                                  0.74012419,
                                  0.82000199,
                                  0.88641553,
                                  0.93827455,
                                  0.97472856,
                                  0.99518722])

        if N == 32:
            weights = np.array([ 0.00701815,
                                 0.01627743,
                                 0.02539101,
                                 0.03427455,
                                 0.04283599,
                                 0.05099787,
                                 0.05868394,
                                 0.06582206,
                                 0.07234561,
                                 0.0781937,
                                 0.08331171,
                                 0.08765187,
                                 0.09117365,
                                 0.09384416,
                                 0.09563848,
                                 0.09653984,
                                 0.09653984,
                                 0.09563848,
                                 0.09384416,
                                 0.09117365,
                                 0.08765187,
                                 0.08331171,
                                 0.0781937,
                                 0.07234561,
                                 0.06582206,
                                 0.05868394,
                                 0.05099787,
                                 0.04283599,
                                 0.03427455,
                                 0.02539101,
                                 0.01627743,
                                 0.00701815])
            abscissas = np.array([-0.99726386,
                                  -0.98561151,
                                  -0.96476226,
                                  -0.93490608,
                                  -0.89632116,
                                  -0.84936761,
                                  -0.7944838,
                                  -0.73218212,
                                  -0.66304427,
                                  -0.58771576,
                                  -0.50689991,
                                  -0.42135128,
                                  -0.3318686,
                                  -0.23928736,
                                  -0.14447196,
                                  -0.04830767,
                                  0.04830767,
                                  0.14447196,
                                  0.23928736,
                                  0.3318686,
                                  0.42135128,
                                  0.50689991,
                                  0.58771576,
                                  0.66304427,
                                  0.73218212,
                                  0.7944838,
                                  0.84936761,
                                  0.89632116,
                                  0.93490608,
                                  0.96476226,
                                  0.98561151,
                                  0.99726386])
        return abscissas, weights
                     
