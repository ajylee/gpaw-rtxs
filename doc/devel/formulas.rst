========
Formulas
========

.. default-role:: math


Coulomb
=======

.. math::

    \frac{1}{|\br-\br'|} =
    \sum_\ell \sum_{m=-\ell}^\ell
    \frac{4\pi}{2\ell+1}
    \frac{r_<^\ell}{r_>^{\ell+1}}
    Y_{\ell m}^*(\hat\br) Y_{\ell m}(\hat\br')

or

.. math::

    \frac{1}{r} = \int d\mathbf{G}\frac{4\pi}{G^2}
    e^{i\mathbf{G}\cdot\br}.


Gaussians
=========

.. math:: n(r) = (\alpha/\pi)^{3/2} e^{-\alpha r^2},

.. math:: \int_0^\infty 4\pi r^2 dr n(r) = 1

Its Fourrier transform is:

.. math::

    n(k) = \int d\br e^{i\mathbf{k}\cdot\br} n(r) =
    \int_0^\infty 4\pi r^2 dr \frac{\sin(kr)}{kr} n(r) =
    e^{-k^2/(4a)}.

With `\nabla^2 v=4\pi n`, we get the potential:

.. math:: v(r) = -\frac{\text{erf}(\sqrt\alpha r)}{r},

and the energy:

.. math::

    \frac12 \int_0^\infty 4\pi r^2 dr n(r) v(r) =
    \sqrt{\frac{\alpha}{2\pi}}.


Hydrogen
========

The 1s orbital:

.. math:: \psi_{\text{1s}}(r) = 2Y_{00} e^{-r},

and the density is:

.. math:: n(r) = |\psi_{\text{1s}}(r)|^2 = e^{-2r}/\pi.


Radial Schrödinger equation
===========================

With `\psi_{n\ell m}(\br) = u(r) / r Y_{\ell m}(\hat\br)`, we have the
radial Schrödinger equation:

.. math::

   -\frac12 \frac{d^2u}{dr^2} + \frac{\ell(\ell + 1)}{2r^2} u + v u
   = \epsilon u.

We want to solve this equation on a non-equidistant radial grid with
`r_g=r(g)` for `g=0,1,...`.  Inserting `u(r) = a(g) r^{\ell + 1}`, we
get:

.. math::

   \frac{d^2 a}{dg^2} (\frac{dg}{dr})^2 r +
   \frac{da}{dg}(r \frac{d^2g}{dr^2} + 2 (\ell + 1) \frac{dg}{dr}) +
   2 r (\epsilon - v) a = 0.


Including Scalar-relativistic corrections
-----------------------------------------

The scalar-relativistic equation is:

.. math::

   -\frac{1}{2 M} \frac{d^2u}{dr^2} + \frac{\ell(\ell + 1)}{2Mr^2} u -
   \frac{1}{(2Mc)^2}\frac{dv}{dr}(\frac{du}{dr}-\frac{u}{r}) + v u
   = \epsilon u.

where the relativistic mass is:

.. math::

   M = 1 - \frac{1}{2c^2} (v - \epsilon).

With `u(r) = a(g) r^{\ell + 1}` and `\kappa = (dv/dr)/(2Mc^2)`:

.. math::

   \frac{d^2 a}{dg^2} (\frac{dg}{dr})^2 r +
   \frac{da}{dg}(r \kappa \frac{dg}{dr} + r \frac{d^2g}{dr^2} +
   2 (\ell + 1) \frac{dg}{dr}) +
   [2 M r (\epsilon - v) + \ell \kappa] a = 0.

