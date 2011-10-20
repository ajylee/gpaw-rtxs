from math import pi

import numpy as np

from gpaw.spline import Spline
from gpaw.utilities import hartree, divrl, _fact as fac


def fsbt(l, fr_g, r_g, G_k):
    """Fast spherical Bessel transform.

    Returns::
       
                 oo
         __ l+1 / 2                l
        4||G    |r dr j (Gr) f(r) r ,
                /      l
                 0

    using l+1 fft's."""

    N = (len(G_k) - 1) * 2
    f_k = 0.0
    for n in range(l + 1):
        f_k += (4 * pi * r_g[1] * (1j)**(l + 1 - n) *
                fac[l + n] / fac[l - n] / fac[n] / 2**n *
                np.fft.rfft(fr_g * r_g**(l - n), N)).real * G_k**(l - n)
    return f_k


class RadialGridDescriptor:
    def __init__(self, r_g, dr_g, default_spline_points=25):
        """Grid descriptor for radial grid."""
        self.r_g = r_g
        self.dr_g = dr_g
        self.N = len(r_g)
        self.dv_g = 4 * pi * r_g**2 * dr_g
        self.default_spline_points = default_spline_points

    def __len__(self):
        return self.N

    def zeros(self, x=()):
        a_xg = self.empty(x)
        a_xg[:] = 0
        return a_xg

    def empty(self, x=()):
        if isinstance(x, int):
            x = (x,)
        return np.zeros(x + (self.N,))

    def integrate(self, a_xg, n=0):
        assert n >= -2
        return np.dot(a_xg[..., 1:],
                      (self.r_g**(2 + n) * self.dr_g)[1:]) * (4 * pi)

    def derivative(self, n_g, dndr_g=None):
        """Finite-difference derivative of radial function."""
        if dndr_g is None:
            dndr_g = self.empty()
        dndr_g[0] = n_g[1] - n_g[0]
        dndr_g[1:-1] = 0.5 * (n_g[2:] - n_g[:-2])
        dndr_g[-1] = n_g[-1] - n_g[-2]
        dndr_g /= self.dr_g
        return dndr_g

    def derivative2(self, a_g, b_g):
        """Finite-difference derivative of radial function.

        For an infinitely dense grid, this method would be identical
        to the `derivative` method."""
        
        c_g = a_g / self.dr_g
        b_g[0] = 0.5 * c_g[1] + c_g[0]
        b_g[1] = 0.5 * c_g[2] - c_g[0]
        b_g[1:-1] = 0.5 * (c_g[2:] - c_g[:-2])
        b_g[-2] = c_g[-1] - 0.5 * c_g[-3]
        b_g[-1] = -c_g[-1] - 0.5 * c_g[-2]

    def interpolate(self, f_g, r_x):
        from scipy.interpolate import InterpolatedUnivariateSpline
        return InterpolatedUnivariateSpline(self.r_g, f_g)(r_x)
        
    def fft(self, fr_g, l=0, N=None):
        """Fourier transform.

        Returns G and f(G) arrays::
           
                                          _ _
               l    ^    / _         ^   iG.r
          f(G)i Y  (G) = |dr f(r)Y  (r) e    .
                 lm      /        lm
        """

        if N is None:
            N = self.N

        assert N % 2 == 0

        r_x = np.linspace(0, self.r_g[-1], N)
        fr_x = self.interpolate(fr_g, r_x)

        G_k = np.linspace(0, pi / r_x[1], N // 2 + 1)
        f_k = fsbt(l, fr_x, r_x, G_k)
        f_k[1:] /= G_k[1:]**(l + 1)
        if l == 0:
            f_k[0] = 4 * pi * np.dot(r_x, fr_x) * r_x[1]
        return G_k, f_k

    def filter(self, f_g, rcut, Gcut, l=0, M=1):
        Rcut = 100.0
        N = 1024 * 8
        r_x = np.linspace(0, Rcut, N, endpoint=False)
        h = Rcut / N

        alpha = 2.0
        mcut = np.exp(-alpha * rcut**2)
        r2_x = r_x**2
        m_x = np.exp(-alpha * r2_x)
        for n in range(M):
            m_x -= (alpha * (rcut**2 - r2_x))**n * (mcut / fac[n])
        xcut = int(np.ceil(rcut / r_x[1]))
        m_x[xcut:] = 0.0

        G_k = np.linspace(0, pi / h, N // 2 + 1)

        fr_x = self.interpolate(f_g * self.r_g**(1-l), r_x)
        fG0_k = fsbt(l, fr_x, r_x, G_k)
        mG_k = fsbt(0, m_x*r_x, r_x, G_k)

        fr_x[:xcut] /= m_x[:xcut]

        fG_k = fsbt(l, fr_x, r_x, G_k)
        kcut = int(Gcut / G_k[1])
        fG_k[kcut:] = 0.0
        fG_k[1:] /= G_k[1:]**(2*l)
        ffr_x = fsbt(l, fG_k, G_k, r_x[:N // 2 + 1])/(4*pi)**2/pi*2
        ffr_x[1:] /= r_x[1:N//2+1]**(2*l)
        import pylab as p
        p.plot(self.r_g, f_g*self.r_g)
        p.plot(r_x[:N // 2 + 1],ffr_x*m_x[:N // 2 + 1])
        p.show()
        
        fG2_k = fsbt(l, ffr_x*m_x[:N // 2 + 1], r_x[:N // 2 + 1], G_k)
        p.plot(G_k,mG_k)
        p.plot(G_k,fG0_k)
        p.plot(G_k,fG2_k)
        p.show()
        return
        l=2
        mG_k = fsbt(l, m_x * r_x, r_x, G_k)
        mG_k[1:]/=G_k[1:]**4
        mr_x = fsbt(l, mG_k, G_k, r_x[:N // 2 + 1])/(4*pi)**2/pi*2
        import pylab as p
        p.plot(r_x, m_x*r_x**5)
        p.plot(r_x[:N // 2 + 1],mr_x)
        p.show()
        
        self.fft(m_g * self.r_g, l)
        from_g = f_g * self.r_g
        from_g[:gcut] /= m_g[:gcut]
        #self.plot(from_g,show=1)
        G_k,f_k = self.fft(from_g, l)
        #p.plot(G_k, m_k)
        #p.plot(G_k, f_k)
        r_x = np.linspace(0, pi / G_k[1], 1024+1)
        fr_x = fsbt(l, f_k * G_k , G_k, r_x)

        r2_x = r_x**2
        m_x = np.exp(-alpha * r2_x)
        for n in range(M):
            m_x -= (alpha * (rcut**2 - r2_x))**n * (mcut / fac[n])

        print self.N, self.r_g[-1],r_x[-1]
        p.plot(r_x, fr_x * m_x / 250)
        p.plot(self.r_g, f_g * self.r_g)
        p.show()

    def purepythonpoisson(self, n_g, l=0):
        r_g = self.r_g
        dr_g = self.dr_g
        a_g = -4 * pi * n_g * r_g * dr_g
        a_g[1:] /= r_g[1:]**l
        A_g = np.add.accumulate(a_g)
        vr_g = self.zeros()
        vr_g[1:] = A_g[:-1] + 0.5 * a_g[1:]
        vr_g -= A_g[-1]
        vr_g *= r_g**(1 + l)
        a_g *= r_g**(2 * l + 1)
        A_g = np.add.accumulate(a_g)
        vr_g[1:] -= A_g[:-1] + 0.5 * a_g[1:]
        vr_g[1:] /= r_g[1:]**l
        return vr_g
    
    def poisson(self, n_g, l=0):  # Old C version
        vr_g = self.zeros()
        nrdr_g = n_g * self.r_g * self.dr_g
        beta = self.a / self.b
        ng = int(round(1.0 / self.b))
        assert abs(ng - 1 / self.b) < 1e-5
        hartree(l, nrdr_g, beta, ng, vr_g)
        #vrp_g = self.purepythonpoisson(n_g,l)
        #assert abs(vr_g-vrp_g).max() < 1e-12
        return vr_g

    def pseudize(self, a_g, gc, l=0, points=4):
        """Construct smooth continuation of a_g for g<gc.
        
        Returns (b_g, c_p) such that b_g=a_g for g >= gc and::
        
                P-1      2(P-1-p)+l
            b = Sum c_p r 
             g  p=0      g

        for g < gc+P.
        """
        assert isinstance(gc, int) and gc > 10
        
        r_g = self.r_g
        i = range(gc, gc + points)
        r_i = r_g[i]
        c_p = np.polyfit(r_i**2, a_g[i] / r_i**l, points - 1)
        b_g = a_g.copy()
        b_g[:gc] = np.polyval(c_p, r_g[:gc]**2) * r_g[:gc]**l
        return b_g, c_p

    def pseudize_normalized(self, a_g, gc, l=0, points=3):
        """Construct normalized smooth continuation of a_g for g<gc.
        
        Returns (b_g, c_p) such that b_g=a_g for g >= gc and::
        
            /        2  /        2
            | dr b(r) = | dr a(r)
            /           /
        """

        b_g = self.pseudize(a_g, gc, l, points)[0]
        c_x = np.empty(points + 1)
        gc0 = gc // 2
        x0 = b_g[gc0]
        r_g = self.r_g
        i = [gc0] + range(gc, gc + points)
        r_i = r_g[i]
        norm = self.integrate(a_g**2)
        def f(x):
            b_g[gc0] = x
            c_x[:] = np.polyfit(r_i**2, b_g[i] / r_i**l, points)
            b_g[:gc] = np.polyval(c_x, r_g[:gc]**2) * r_g[:gc]**l
            return self.integrate(b_g**2) - norm
        from scipy.optimize import fsolve
        fsolve(f, x0)
        return b_g, c_x
        
    def plot(self, a_g, n=0, rc=4.0, show=False):
        import matplotlib.pyplot as plt
        r_g = self.r_g[:len(a_g)]
        if n < 0:
            r_g = r_g[1:]
            a_g = a_g[1:]
        plt.plot(r_g, a_g * r_g**n)
        plt.axis(xmax=rc)
        if show:
            plt.show()

    def floor(self, r):
        return np.floor(self.r2g(r)).astype(int)
    
    def round(self, r):
        return np.around(self.r2g(r)).astype(int)
    
    def ceil(self, r):
        return np.ceil(self.r2g(r)).astype(int)

    def spline(self, a_g, rcut, l=0, points=None):
        if points is None:
            points = self.default_spline_points

        b_g = a_g.copy()
        N = len(b_g)
        if l > 0:
            b_g = divrl(b_g, l, self.r_g[:N])
            #b_g[1:] /= self.r_g[1:]**l
            #b_g[0] = b_g[1]
            
        r_i = np.linspace(0, rcut, points + 1)
        #g_i = np.clip(self.ceil(r_i), 1, self.N - 2)
        #g_i = np.clip(self.round(r_i), 1, self.N - 2)
        g_i = np.clip((self.r2g(r_i)+0.5).astype(int), 1, N - 2)
        if 0:#a_g[0] < 0:
            print a_g[[0,1,2,-10,-2,-1]]
            print rcut,l,points, len(a_g)
            print g_i;dcvg
        r1_i = self.r_g[g_i - 1]
        r2_i = self.r_g[g_i]
        r3_i = self.r_g[g_i + 1]
        x1_i = (r_i - r2_i) * (r_i - r3_i) / (r1_i - r2_i) / (r1_i - r3_i)
        x2_i = (r_i - r1_i) * (r_i - r3_i) / (r2_i - r1_i) / (r2_i - r3_i)
        x3_i = (r_i - r1_i) * (r_i - r2_i) / (r3_i - r1_i) / (r3_i - r2_i)
        b1_i = b_g[g_i - 1]
        b2_i = b_g[g_i]
        b3_i = b_g[g_i + 1]
        b_i = b1_i * x1_i + b2_i * x2_i + b3_i * x3_i
        return Spline(l, rcut, b_i)


class EquidistantRadialGridDescriptor(RadialGridDescriptor):
    def __init__(self, h, N=1000, h0=0.0):
        """Equidistant radial grid descriptor.

        The radial grid is r(g) = h0 + g*h,  g = 0, 1, ..., N - 1."""

        RadialGridDescriptor.__init__(self, h * np.arange(N) + h0, h)

    def r2g(self, r):
        return (r - self.r_g[0]) / (self.r_g[1] - self.r_g[0])

    def spline(self, a_g, l=0):
        b_g = a_g.copy()
        if l > 0:
            b_g = divrl(b_g, l, self.r_g[:len(a_g)])
            #b_g[1:] /= self.r_g[1:]**l
            #b_g[0] = b_g[1]
        return Spline(l, self.r_g[len(a_g) - 1], b_g)


class AERadialGridDescriptor(RadialGridDescriptor):
    def __init__(self, a, b, N=1000, default_spline_points=25):
        """Radial grid descriptor for all-electron calculation.

        The radial grid is::

                     a g
            r(g) = -------,  g = 0, 1, ..., N - 1
                   1 - b g
        """

        self.a = a
        self.b = b
        g = np.arange(N)
        r_g = self.a * g / (1 - self.b * g)
        dr_g = (self.b * r_g + self.a)**2 / self.a
        RadialGridDescriptor.__init__(self, r_g, dr_g, default_spline_points)
                                      

    def r2g(self, r):
        # return r / (r * self.b + self.a)
        # Hack to preserve backwards compatibility:
        ng = 1.0 / self.b
        beta = self.a / self.b
        return ng * r / (beta + r)

    def xml(self, id='grid1'):
        if abs(self.N - 1 / self.b) < 1e-5:
            return (('<radial_grid eq="r=a*i/(n-i)" a="%r" n="%d" ' +
                     'istart="0" iend="%d" id="%s"/>') % 
                    (self.a * self.N, self.N, self.N - 1, id))
        return (('<radial_grid eq="r=a*i/(1-b*i)" a="%r" b="%r" n="%d" ' +
                 'istart="0" iend="%d" id="%s"/>') % 
                (self.a, self.b, self.N, self.N - 1, id))

    def d2gdr2(self):
        return -2 * self.a * self.b / (self.b * self.r_g + self.a)**3
