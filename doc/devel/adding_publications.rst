.. _adding_publications:

=========================================
Adding papers to the list of publications
=========================================

The GPAW related publications appearing in the 
:ref:`literature` section, please edit the file 
:file:`literature.rst` in the :file:`doc/documention` 
directory.

Use the form::

  #) First Author, Second Author, and Final Author
  
     `Title of the paper`__

     \J. Chem. Phys. 134, 134109 (2011)

     __ http://dx.doi.org/10.1063/1.3574836

     .. XX May 2011

Escape the starting single letters in author lines and in
journal names with ``\`` (e.g. ``\A. Einstein``,  ``\J. Chem``) in order
to avoid starting an enumerated list.

The date ``XX May 2011`` should be the publication date, and the publications
in :file:`literature.rst` should be in chronological order according
to publication date. It is recommened to use  `DOIs <http://www.doi.org/>`_ 
and `http://dx.doi.org <http://dx.doi.org>`_ for links to the publications.
