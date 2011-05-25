.. _bandstructures:

=========================================
Calculation of electronic band structures
=========================================

The script :svn:`~doc/tutorials/bandstructures/bandstructure.py` will
calculate the band structure of Na along the Gamma-X direction.  The
resulting band structure is shown below.

.. figure:: sodium_bands.png

(This plot was made using :svn:`~doc/tutorials/bandstructures/plot_bands.py`)

One should note that as GPAW only works with orthorhombic cells, the
unit cell here is larger than the primitive BCC cell.  Accordingly,
the Brillouin zone is smaller, and bands are folded back from the
larger primitive Brillouin zone. For a description of the symmetry
labels of the Brillouin zone; see the figure below.

.. figure:: ../../_static/bz-all.png
   :width: 600 px
   :align: left
