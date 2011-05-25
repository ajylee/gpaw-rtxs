/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "spline.h"
#include "lfc.h"
#include "bmgs/spherical_harmonics.h"
#include "bmgs/bmgs.h"


static void lfc_dealloc(LFCObject *self)
{
  if (self->bloch_boundary_conditions)
    free(self->phase_i);
  free(self->volume_i);
  free(self->work_gm);
  free(self->ngm_W);
  free(self->i_W);
  free(self->volume_W);
  PyObject_DEL(self);
}

PyObject* calculate_potential_matrix(LFCObject *self, PyObject *args);
PyObject* integrate(LFCObject *self, PyObject *args);
PyObject* derivative(LFCObject *self, PyObject *args);
PyObject* normalized_derivative(LFCObject *self, PyObject *args);
PyObject* construct_density(LFCObject *self, PyObject *args);
PyObject* construct_density1(LFCObject *self, PyObject *args);
PyObject* ae_valence_density_correction(LFCObject *self, PyObject *args);
PyObject* ae_core_density_correction(LFCObject *self, PyObject *args);
PyObject* lcao_to_grid(LFCObject *self, PyObject *args);
PyObject* add(LFCObject *self, PyObject *args);
PyObject* calculate_potential_matrix_derivative(LFCObject *self, 
                                                PyObject *args);
PyObject* second_derivative(LFCObject *self, PyObject *args);
PyObject* add_derivative(LFCObject *self, PyObject *args);

static PyMethodDef lfc_methods[] = {
    {"calculate_potential_matrix",
     (PyCFunction)calculate_potential_matrix, METH_VARARGS, 0},
    {"integrate",
     (PyCFunction)integrate, METH_VARARGS, 0},
    {"derivative",
     (PyCFunction)derivative, METH_VARARGS, 0},
    {"normalized_derivative",
     (PyCFunction)normalized_derivative, METH_VARARGS, 0},
    {"construct_density",
     (PyCFunction)construct_density, METH_VARARGS, 0},
    {"construct_density1",
     (PyCFunction)construct_density1, METH_VARARGS, 0},
    {"ae_valence_density_correction",
     (PyCFunction)ae_valence_density_correction, METH_VARARGS, 0},
    {"ae_core_density_correction",
     (PyCFunction)ae_core_density_correction, METH_VARARGS, 0},
    {"lcao_to_grid",
     (PyCFunction)lcao_to_grid, METH_VARARGS, 0},
    {"add",
     (PyCFunction)add, METH_VARARGS, 0},
    {"calculate_potential_matrix_derivative",
     (PyCFunction)calculate_potential_matrix_derivative, METH_VARARGS, 0},
    {"second_derivative",
     (PyCFunction)second_derivative, METH_VARARGS, 0},
    {"add_derivative",
     (PyCFunction)add_derivative, METH_VARARGS, 0},
#ifdef PARALLEL
    {"broadcast",
     (PyCFunction)localized_functions_broadcast, METH_VARARGS, 0},
#endif
    {NULL, NULL, 0, NULL}
};

static PyObject* lfc_getattr(PyObject *obj, char *name)
{
  return Py_FindMethod(lfc_methods, obj, name);
}

static PyTypeObject LFCType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "LocalizedFunctionsCollection",
  sizeof(LFCObject),
  0,
  (destructor)lfc_dealloc,
  0,
  lfc_getattr
};

PyObject * NewLFCObject(PyObject *obj, PyObject *args)
{
  PyObject* A_Wgm_obj;
  const PyArrayObject* M_W_obj;
  const PyArrayObject* G_B_obj;
  const PyArrayObject* W_B_obj;
  double dv;
  const PyArrayObject* phase_kW_obj;

  if (!PyArg_ParseTuple(args, "OOOOdO",
                        &A_Wgm_obj, &M_W_obj, &G_B_obj, &W_B_obj, &dv,
                        &phase_kW_obj))
    return NULL; 

  LFCObject *self = PyObject_NEW(LFCObject, &LFCType);
  if (self == NULL)
    return NULL;

  self->dv = dv;

  const int* M_W = (const int*)M_W_obj->data;
  self->G_B = (int*)G_B_obj->data;
  self->W_B = (int*)W_B_obj->data;

  if (phase_kW_obj->dimensions[0] > 0) {
    self->bloch_boundary_conditions = true;
    self->phase_kW = (double complex*)phase_kW_obj->data;
  }
  else {
    self->bloch_boundary_conditions = false;
  }

  int nB = G_B_obj->dimensions[0];
  int nW = PyList_Size(A_Wgm_obj);

  self->nW = nW;
  self->nB = nB;

  int nimax = 0;
  int ngmax = 0;
  int ni = 0;
  int Ga = 0;
  for (int B = 0; B < nB; B++) {
    int Gb = self->G_B[B];
    int nG = Gb - Ga;
    if (ni > 0 && nG > ngmax)
      ngmax = nG;
    if (self->W_B[B] >= 0)
      ni += 1;
    else {
      if (ni > nimax)
        nimax = ni;
      ni--;
    }
    Ga = Gb;
  }
  assert(ni == 0);
  
  self->volume_W = GPAW_MALLOC(LFVolume, nW);
  self->i_W = GPAW_MALLOC(int, nW);
  self->ngm_W = GPAW_MALLOC(int, nW);

  int nmmax = 0;
  for (int W = 0; W < nW; W++) {
    const PyArrayObject* A_gm_obj = \
      (const PyArrayObject*)PyList_GetItem(A_Wgm_obj, W);
    LFVolume* volume = &self->volume_W[W];
    volume->A_gm = (const double*)A_gm_obj->data;
    self->ngm_W[W] = A_gm_obj->dimensions[0] * A_gm_obj->dimensions[1];
    volume->nm = A_gm_obj->dimensions[1];
    volume->M = M_W[W];
    volume->W = W;
    if (volume->nm > nmmax)
      nmmax = volume->nm;
  }
  self->work_gm = GPAW_MALLOC(double, ngmax * nmmax);
  self->volume_i = GPAW_MALLOC(LFVolume, nimax);
  if (self->bloch_boundary_conditions)
    self->phase_i = GPAW_MALLOC(complex double, nimax);
  
  return (PyObject*)self;
}

PyObject* calculate_potential_matrix(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* vt_G_obj;
  PyArrayObject* Vt_MM_obj;
  int k;
  int Mstart;
  int Mstop;

  if (!PyArg_ParseTuple(args, "OOiii", &vt_G_obj, &Vt_MM_obj, &k,
                        &Mstart, &Mstop))
    return NULL; 

  const double* vt_G = (const double*)vt_G_obj->data;

  int nM = Vt_MM_obj->dimensions[1];
  double dv = lfc->dv;
  double* work_gm = lfc->work_gm;
  if (!lfc->bloch_boundary_conditions) {
    double* Vt_MM = (double*)Vt_MM_obj->data;
    GRID_LOOP_START(lfc, -1) { // ORDINARY/GAMMA-POINT
      for (int i1 = 0; i1 < ni; i1++) {
	LFVolume* v1 = volume_i + i1;
	int M1 = v1->M;
	int nm1 = v1->nm;
	int M1p = MAX(M1, Mstart);
	int nm1p = MIN(M1 + nm1, Mstop) - M1p;
	if (nm1p <= 0)
	  continue;
	int gm = M1p - M1;
	int gm1 = 0;
	const double* A1_gm = v1->A_gm;
	for (int G = Ga; G < Gb; G++, gm += nm1 - nm1p) {
	  double vtdv = vt_G[G] * dv;
	  for (int m1 = 0; m1 < nm1p; m1++, gm1++, gm++)
	    work_gm[gm1] = vtdv * A1_gm[gm];
	}
	for (int i2 = 0; i2 < ni; i2++) {
	  LFVolume* v2 = volume_i + i2;
	  int M2 = v2->M;
	  if (M1 >= M2) {
	    int nm2 = v2->nm;
	    const double* A2_gm = v2->A_gm;
	    double* Vt_mm = Vt_MM + (M1p - Mstart) * nM + M2;
	    for (int g = 0; g < nG; g++){
	      int gnm1 = g * nm1p;
	      int gnm2 = g * nm2;
	      for (int m1 = 0; m1 < nm1p; m1++) {
		int m1nM = m1 * nM;
		for (int m2 = 0; m2 < nm2; m2++)
		  Vt_mm[m2 + m1nM] += A2_gm[gnm2 + m2] * work_gm[gnm1 + m1];
	      }
	    }
	  }
	}
      }
    }
    GRID_LOOP_STOP(lfc, -1);
  }
  else {
    complex double* Vt_MM = (complex double*)Vt_MM_obj->data;
    GRID_LOOP_START(lfc, k) {  // KPOINT CALC POT MATRIX
      for (int i1 = 0; i1 < ni; i1++) {
        LFVolume* v1 = volume_i + i1;
        double complex conjphase1 = conj(phase_i[i1]);
        int M1 = v1->M;
        int nm1 = v1->nm;
	int M1p = MAX(M1, Mstart);
	int nm1p = MIN(M1 + nm1, Mstop) - M1p;
	if (nm1p <= 0)
	  continue;
	int gm = M1p - M1;
        int gm1 = 0;
        const double* A1_gm = v1->A_gm;
        for (int G = Ga; G < Gb; G++, gm += nm1 - nm1p) {
          double vtdv = vt_G[G] * dv;
          for (int m1 = 0; m1 < nm1p; m1++, gm1++, gm++)
            work_gm[gm1] = vtdv * A1_gm[gm];
        }
        for (int i2 = 0; i2 < ni; i2++) {
          LFVolume* v2 = volume_i + i2;
          const double* A2_gm = v2->A_gm;
          int M2 = v2->M;
          if (M1 >= M2) {
            int nm2 = v2->nm;
            double complex phase = conjphase1 * phase_i[i2];
            double complex* Vt_mm = Vt_MM + (M1p - Mstart) * nM + M2;
            for (int g = 0; g < nG; g++) {
              int gnm1 = g * nm1p;
              int gnm2 = g * nm2;
              int m1nM = 0;
              for (int m1 = 0; m1 < nm1p; m1++, m1nM += nM) {
                complex double wphase = work_gm[gnm1 + m1] * phase;
                for (int m2 = 0; m2 < nm2; m2++) {
                  Vt_mm[m1nM + m2] += A2_gm[gnm2 + m2] * wphase;
                }
              }
            }
          }
        }
      }
    }
    GRID_LOOP_STOP(lfc, k);
  }
  Py_RETURN_NONE;
}

PyObject* integrate(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* a_xG_obj;
  PyArrayObject* c_xM_obj;
  int q;

  if (!PyArg_ParseTuple(args, "OOi", &a_xG_obj, &c_xM_obj, &q))
    return NULL; 

  int nd = a_xG_obj->nd;
  npy_intp* dims = a_xG_obj->dimensions;
  int nx = PyArray_MultiplyList(dims, nd - 3);
  int nG = PyArray_MultiplyList(dims + nd - 3, 3);
  int nM = c_xM_obj->dimensions[c_xM_obj->nd - 1];
  double dv = lfc->dv;

  if (!lfc->bloch_boundary_conditions) {
    const double* a_G = (const double*)a_xG_obj->data;
    double* c_M = (double*)c_xM_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, -1) {
        for (int i = 0; i < ni; i++) {
          LFVolume* v = volume_i + i;
          const double* A_gm = v->A_gm;
          int nm = v->nm;
          double* c_M1 = c_M + v->M;
          for (int gm = 0, G = Ga; G < Gb; G++){
            double av = a_G[G] * dv;
            for (int m = 0; m < nm; m++, gm++){
              c_M1[m] += av * A_gm[gm];
            }
          }
        }
      }
      GRID_LOOP_STOP(lfc, -1);
      c_M += nM;
      a_G += nG;
    }
  }
  else {
    const complex double* a_G = (const complex double*)a_xG_obj->data;
    complex double* c_M = (complex double*)c_xM_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, q) {
        for (int i = 0; i < ni; i++) {
          LFVolume* v = volume_i + i;
          int nm = v->nm;
          complex double* c_M1 = c_M + v->M;
          const double* A_gm = v->A_gm;
          double complex vphase = phase_i[i] * dv;
          for (int gm = 0, G = Ga; G < Gb; G++){
            double complex avphase = a_G[G] * vphase;
            for (int m = 0; m < nm; m++, gm++){
              c_M1[m] += avphase * A_gm[gm];
            }
          }
        }
      }
      GRID_LOOP_STOP(lfc, q);
      c_M += nM;
      a_G += nG;
    }
  }
  Py_RETURN_NONE;
}

PyObject* construct_density(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* rho_MM_obj;
  PyArrayObject* nt_G_obj;
  int k;
  int Mstart, Mstop;

  if (!PyArg_ParseTuple(args, "OOiii", &rho_MM_obj, &nt_G_obj, &k,
                        &Mstart, &Mstop))
    return NULL; 
  
  double* nt_G = (double*)nt_G_obj->data;
  
  int nM = rho_MM_obj->dimensions[1];
  
  double* work_gm = lfc->work_gm;

  if (!lfc->bloch_boundary_conditions) {
    const double* rho_MM = (const double*)rho_MM_obj->data;
    GRID_LOOP_START(lfc, -1) {
      for (int i1 = 0; i1 < ni; i1++) {
        LFVolume* v1 = volume_i + i1;
        int M1 = v1->M;
        int nm1 = v1->nm;

	int M1p = MAX(M1, Mstart);
	int nm1p = MIN(M1 + nm1, Mstop) - M1p;
	if (nm1p <= 0)
	  continue;

        memset(work_gm, 0, nG * nm1 * sizeof(double));
        double factor = 1.0;

        int m1end = MIN(nm1, Mstop - M1);
        int m1start = MAX(0, Mstart - M1);

        for (int i2 = i1; i2 < ni; i2++) {
          LFVolume* v2 = volume_i + i2;
          int M2 = v2->M;
          int nm2 = v2->nm;
          const double* rho_mm = rho_MM + (M1p - Mstart) * nM + M2;
          //assert(M1 - Mstart + m1start >= 0);
          for (int g = 0; g < nG; g++) {
            for (int m1 = m1start, m1p = 0; m1 < m1end; m1++, m1p++) {
              for (int m2 = 0; m2 < nm2; m2++) {
                work_gm[g * nm1 + m1] += (v2->A_gm[g * nm2 + m2] * 
                                          rho_mm[m1p * nM + m2] *
                                          factor);
              }
            }
          }
          factor = 2.0;
        }
        int gm1 = 0;
        for (int G = Ga; G < Gb; G++) {
          double nt = 0.0;
          for (int m1 = 0; m1 < nm1; m1++, gm1++) {
            nt += v1->A_gm[gm1] * work_gm[gm1];
          }
          nt_G[G] += nt;
        }
      }
    }
    GRID_LOOP_STOP(lfc, -1);
  }
  else {
    const double complex* rho_MM = (const double complex*)rho_MM_obj->data;
    GRID_LOOP_START(lfc, k) {
      for (int i1 = 0; i1 < ni; i1++) {
        LFVolume* v1 = volume_i + i1;
        int M1 = v1->M;
        int nm1 = v1->nm;

	int M1p = MAX(M1, Mstart);
	int nm1p = MIN(M1 + nm1, Mstop) - M1p;
	if (nm1p <= 0)
	  continue;

        memset(work_gm, 0, nG * nm1 * sizeof(double));
        double complex factor = 1.0;
	
	int m1end = MIN(nm1, Mstop - M1);
	int m1start = MAX(0, Mstart - M1);
	
        for (int i2 = i1; i2 < ni; i2++) {
          if (i2 > i1)
            factor = 2.0 * phase_i[i1] * conj(phase_i[i2]);
          
          double rfactor = creal(factor);
          double ifactor = cimag(factor);
          
          LFVolume* v2 = volume_i + i2;
          const double* A2_gm = v2->A_gm;
          int M2 = v2->M;
          int nm2 = v2->nm;
          const double complex* rho_mm = rho_MM + (M1p - Mstart) * nM + M2;
          double rrho, irho, rwork, iwork;
          complex double rho;
          for (int g = 0; g < nG; g++) {
            int gnm1 = g * nm1;
            int gnm2 = g * nm2;
            int m1pnM = 0;
            for (int m1 = m1start, m1p=0; m1 < m1end; m1++, m1p++) {
              m1pnM = m1p * nM;
              iwork = 0;
              rwork = 0;
              for (int m2 = 0; m2 < nm2; m2++) {
                rho = rho_mm[m1pnM + m2];
                rrho = creal(rho);
                irho = cimag(rho);
                rwork += A2_gm[gnm2 + m2] * rrho;
                iwork += A2_gm[gnm2 + m2] * irho;
                // We could save one of those multiplications if the buffer
                // were twice as large

                //work += A2_gm[gnm2 + m2] * (rfactor * rrho - ifactor * irho);
              }
              //work_gm[m1 + gnm1] += work;
              work_gm[m1 + gnm1] += rwork * rfactor - iwork * ifactor;
            }
          }
        }
        int gm1 = 0;
        const double* A1_gm = v1->A_gm;
        for (int G = Ga; G < Gb; G++) {
          double nt = 0.0;
          for (int m1 = 0; m1 < nm1; m1++, gm1++) {
            nt += A1_gm[gm1] * work_gm[gm1];
          }
          nt_G[G] += nt;
        }
      }
    }
    GRID_LOOP_STOP(lfc, k);
  }
  Py_RETURN_NONE;
}

PyObject* construct_density1(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* f_M_obj;
  PyArrayObject* nt_G_obj;
  
  if (!PyArg_ParseTuple(args, "OO", &f_M_obj, &nt_G_obj))
    return NULL; 
  
  const double* f_M = (const double*)f_M_obj->data;
  double* nt_G = (double*)nt_G_obj->data;

  GRID_LOOP_START(lfc, -1) {
    for (int i = 0; i < ni; i++) {
      LFVolume* v = volume_i + i;
      for (int gm = 0, G = Ga; G < Gb; G++) {
        for (int m = 0; m < v->nm; m++, gm++) {
          nt_G[G] += v->A_gm[gm] * v->A_gm[gm] * f_M[v->M + m];
        }
      }
    }
  }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}

PyObject* lcao_to_grid(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* c_M_obj;
  PyArrayObject* psit_G_obj;
  int k;

  if (!PyArg_ParseTuple(args, "OOi", &c_M_obj, &psit_G_obj, &k))
    return NULL; 
  
  if (!lfc->bloch_boundary_conditions) {
    if (c_M_obj->descr->type_num == PyArray_DOUBLE) {
      const double* c_M = (const double*)c_M_obj->data;
      double* psit_G = (double*)psit_G_obj->data;
      GRID_LOOP_START(lfc, -1) {
        for (int i = 0; i < ni; i++) {
          LFVolume* v = volume_i + i;
          for (int gm = 0, G = Ga; G < Gb; G++) {
            for (int m = 0; m < v->nm; m++, gm++) {
              psit_G[G] += v->A_gm[gm] * c_M[v->M + m];
            }
          }
        }
      }
      GRID_LOOP_STOP(lfc, -1);
    }
    else {
      const double complex* c_M = (const double complex*)c_M_obj->data;
      double complex* psit_G = (double complex*)psit_G_obj->data;
      GRID_LOOP_START(lfc, -1) {
        for (int i = 0; i < ni; i++) {
          LFVolume* v = volume_i + i;
          for (int gm = 0, G = Ga; G < Gb; G++) {
            for (int m = 0; m < v->nm; m++, gm++) {
              psit_G[G] += v->A_gm[gm] * c_M[v->M + m];
            }
          }
        }
      }
      GRID_LOOP_STOP(lfc, -1);
    }
  }
  else {
    const double complex* c_M = (const double complex*)c_M_obj->data;
    double complex* psit_G = (double complex*)psit_G_obj->data;
    GRID_LOOP_START(lfc, k) {
      for (int i = 0; i < ni; i++) {
        LFVolume* v = volume_i + i;
        double complex conjphase = conj(phase_i[i]);
        const double* A_gm = v->A_gm;
        const double complex* c_M1 = c_M + v->M;
        for (int gm = 0, G = Ga; G < Gb; G++) {
          double complex psit = 0.0;
          for (int m = 0; m < v->nm; m++, gm++) {
            psit += A_gm[gm] * c_M1[m];
          }
          psit_G[G] += psit * conjphase;
        }
      }
    }
    GRID_LOOP_STOP(lfc, k);
  }
  Py_RETURN_NONE;
}

PyObject* add(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* c_xM_obj;
  PyArrayObject* a_xG_obj;
  int q;

  if (!PyArg_ParseTuple(args, "OOi", &c_xM_obj, &a_xG_obj, &q))
    return NULL; 

  int nd = a_xG_obj->nd;
  npy_intp* dims = a_xG_obj->dimensions;
  int nx = PyArray_MultiplyList(dims, nd - 3);
  int nG = PyArray_MultiplyList(dims + nd - 3, 3);
  int nM = c_xM_obj->dimensions[c_xM_obj->nd - 1];

  if (!lfc->bloch_boundary_conditions) {
    const double* c_M = (const double*)c_xM_obj->data;
    double* a_G = (double*)a_xG_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, -1) {
        for (int i = 0; i < ni; i++) {
          LFVolume* v = volume_i + i;
          for (int gm = 0, G = Ga; G < Gb; G++) {
            for (int m = 0; m < v->nm; m++, gm++) {
              a_G[G] += v->A_gm[gm] * c_M[v->M + m];
            }
          }
        }
      }
      GRID_LOOP_STOP(lfc, -1);
      c_M += nM;
      a_G += nG;
    }
  }
  else {
    const double complex* c_M = (const double complex*)c_xM_obj->data;
    double complex* a_G = (double complex*)a_xG_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, q) {
        for (int i = 0; i < ni; i++) {
          double complex conjphase = conj(phase_i[i]);
          LFVolume* v = volume_i + i;
          const double complex* c_M1 = c_M + v->M;
          const double* A_gm = v->A_gm;
          for (int gm = 0, G = Ga; G < Gb; G++) {
            double complex a = 0.0;
            for (int m = 0; m < v->nm; m++, gm++) {
              a += A_gm[gm] * c_M1[m];
            }
            a_G[G] += a * conjphase;
          }
        }
      }
      GRID_LOOP_STOP(lfc, q);
      c_M += nM;
      a_G += nG;
    }
  }
  Py_RETURN_NONE;
}

PyObject* spline_to_grid(PyObject *self, PyObject *args)
{
  const SplineObject* spline_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* end_c_obj;
  PyArrayObject* pos_v_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyArrayObject* gdcorner_c_obj;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &spline_obj,
                        &beg_c_obj, &end_c_obj, &pos_v_obj, &h_cv_obj,
                        &n_c_obj, &gdcorner_c_obj))
    return NULL; 

  const bmgsspline* spline = (const bmgsspline*)(&(spline_obj->spline));
  long* beg_c = LONGP(beg_c_obj);
  long* end_c = LONGP(end_c_obj);
  double* pos_v = DOUBLEP(pos_v_obj);
  double* h_cv = DOUBLEP(h_cv_obj);
  long* n_c = LONGP(n_c_obj);
  long* gdcorner_c = LONGP(gdcorner_c_obj);

  int l = spline_obj->spline.l;
  int nm = 2 * l + 1;
  double rcut = spline->dr * spline->nbins;

  int ngmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]) *
               (end_c[2] - beg_c[2]));
  double* A_gm = GPAW_MALLOC(double, ngmax * nm);
  
  int nBmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]));
  int* G_B = GPAW_MALLOC(int, 2 * nBmax);

  int nB = 0;
  int ngm = 0;
  int G = -gdcorner_c[2] + n_c[2] * (beg_c[1] - gdcorner_c[1] + n_c[1] 
                    * (beg_c[0] - gdcorner_c[0]));

  for (int g0 = beg_c[0]; g0 < end_c[0]; g0++) {
    for (int g1 = beg_c[1]; g1 < end_c[1]; g1++) {
      int g2_beg = -1; // function boundary coordinates
      int g2_end = -1;
      for (int g2 = beg_c[2]; g2 < end_c[2]; g2++) {
        double x = h_cv[0] * g0 + h_cv[3] * g1 + h_cv[6] * g2 - pos_v[0];
        double y = h_cv[1] * g0 + h_cv[4] * g1 + h_cv[7] * g2 - pos_v[1];
        double z = h_cv[2] * g0 + h_cv[5] * g1 + h_cv[8] * g2 - pos_v[2];
        double r2 = x * x + y * y + z * z;
        double r = sqrt(r2);
        if (r < rcut) {
          if (g2_beg < 0)
            g2_beg = g2; // found boundary
          g2_end = g2;
          double A = bmgs_splinevalue(spline, r);
          double* p = A_gm + ngm;
          
          spherical_harmonics(l, A, x, y, z, r2, p);
          
          ngm += nm;
        }
      }
      if (g2_end >= 0) {
        g2_end++;
        G_B[nB++] = G + g2_beg;
        G_B[nB++] = G + g2_end;
      }
      G += n_c[2];
    }
    G += n_c[2] * (n_c[1] - end_c[1] + beg_c[1]);
  }
  npy_intp gm_dims[2] = {ngm / (2 * l + 1), 2 * l + 1};
  PyArrayObject* A_gm_obj = (PyArrayObject*)PyArray_SimpleNew(2, gm_dims, 
                                                              NPY_DOUBLE);
  
  memcpy(A_gm_obj->data, A_gm, ngm * sizeof(double));
  free(A_gm);
  
  npy_intp B_dims[1] = {nB};
  PyArrayObject* G_B_obj = (PyArrayObject*)PyArray_SimpleNew(1, B_dims,
                                                             NPY_INT);
  memcpy(G_B_obj->data, G_B, nB * sizeof(int));
  free(G_B);

  // PyObjects created in the C code will be initialized with a refcount
  // of 1, for which reason we'll have to decref them when done here
  PyObject* values = Py_BuildValue("(OO)", A_gm_obj, G_B_obj);
  Py_DECREF(A_gm_obj);
  Py_DECREF(G_B_obj);
  return values;
}


// Horrible copy-paste of calculate_potential_matrix
// Surely it must be possible to find a way to actually reuse code
// Maybe some kind of preprocessor thing
PyObject* calculate_potential_matrix_derivative(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* vt_G_obj;
  PyArrayObject* DVt_MM_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  int k, c;
  PyArrayObject* spline_obj_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;
  int Mstart, Mstop;

  if (!PyArg_ParseTuple(args, "OOOOiiOOOii", &vt_G_obj, &DVt_MM_obj, 
                        &h_cv_obj, &n_c_obj, &k, &c,
                        &spline_obj_M_obj, &beg_c_obj,
                        &pos_Wc_obj, &Mstart, &Mstop))
    return NULL;

  const double* vt_G = (const double*)vt_G_obj->data;
  const double* h_cv = (const double*)h_cv_obj->data;
  const long* n_c = (const long*)n_c_obj->data;
  const SplineObject** spline_obj_M = \
    (const SplineObject**)spline_obj_M_obj->data;
  const double (*pos_Wc)[3] = (const double (*)[3])pos_Wc_obj->data;

  long* beg_c = LONGP(beg_c_obj);
  int nM = DVt_MM_obj->dimensions[1];
  double* work_gm = lfc->work_gm;
  double dv = lfc->dv;

  if (!lfc->bloch_boundary_conditions) {
    double* DVt_MM = (double*)DVt_MM_obj->data;
    {
      GRID_LOOP_START(lfc, -1) {
        // In one grid loop iteration, only z changes.
        int iza = Ga % n_c[2] + beg_c[2];
        int iy = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int ix = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        int iz = iza;

        //assert(Ga == ((ix - beg_c[0]) * n_c[1] + (iy - beg_c[1])) 
        //       * n_c[2] + iza - beg_c[2]);

        for (int i1 = 0; i1 < ni; i1++) {
          iz = iza;
          LFVolume* v1 = volume_i + i1;
          int M1 = v1->M;
          const SplineObject* spline_obj = spline_obj_M[M1];
          const bmgsspline* spline = \
            (const bmgsspline*)(&(spline_obj->spline));
          
          int nm1 = v1->nm;

          int M1p = MAX(M1, Mstart);
          int nm1p = MIN(M1 + nm1, Mstop) - M1p;
          if (nm1p <= 0)
            continue;

          double fdYdc_m[nm1];
          double rlYdfdr_m[nm1];
          double f, dfdr;
          int l = (nm1 - 1) / 2;
          const double* pos_c = pos_Wc[v1->W];
          //assert(2 * l + 1 == nm1);
          //assert(spline_obj->spline.l == l);
          int gm1 = 0;
          for (int G = Ga; G < Gb; G++, iz++) {
            double x = h_cv[0] * ix + h_cv[3] * iy + h_cv[6] * iz - pos_c[0];
            double y = h_cv[1] * ix + h_cv[4] * iy + h_cv[7] * iz - pos_c[1];
            double z = h_cv[2] * ix + h_cv[5] * iy + h_cv[8] * iz - pos_c[2];
            double vtdv = vt_G[G] * dv;

            double R_c[] = {x, y, z};
            
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double Rcinvr = r > 1e-15 ? R_c[c] / r : 0.0;
            //assert(G == ((ix - beg_c[0]) * n_c[1] + 
            //             (iy - beg_c[1])) * n_c[2] + iz - beg_c[2]);

            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            //assert (r <= spline->dr * spline->nbins); // important

            switch(c) {
            case 0:
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 1:
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 2:
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdYdc_m);
              break;
            }
            spherical_harmonics(l, dfdr * Rcinvr, x, y, z, r2, rlYdfdr_m);

            int m1start = M1 < Mstart ? nm1 - nm1p : 0;
            for (int m1 = 0; m1 < nm1p; m1++, gm1++) {
              work_gm[gm1] = vtdv * (fdYdc_m[m1 + m1start] 
                                     + rlYdfdr_m[m1 + m1start]);
            }            
          } // end loop over G
          for (int i2 = 0; i2 < ni; i2++) {
            LFVolume* v2 = volume_i + i2;
            int M2 = v2->M;
            const double* A2_start_gm = v2->A_gm;
            const double* A2_gm;
            int nm2 = v2->nm;
            double* DVt_start_mm = DVt_MM + (M1p - Mstart) * nM + M2;
            double* DVt_mm;
            double work;
            for (int g = 0; g < nG; g++) {
              A2_gm = A2_start_gm + g * nm2;
              for (int m1 = 0; m1 < nm1p; m1++) {
                work = work_gm[g * nm1p + m1];
                DVt_mm = DVt_start_mm + m1 * nM;
                for (int m2 = 0; m2 < nm2; m2++) {
                  DVt_mm[m2] += A2_gm[m2] * work;
                }
              }
            }
          } // i2 loop
        } // G loop
      } // i1 loop
      GRID_LOOP_STOP(lfc, -1);
    } // c loop

  }
  else {
    complex double* DVt_MM = (complex double*)DVt_MM_obj->data;
    {
      GRID_LOOP_START(lfc, k) {
        // In one grid loop iteration, only z changes.
        int iza = Ga % n_c[2] + beg_c[2];
        int iy = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int ix = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        int iz = iza;

        for (int i1 = 0; i1 < ni; i1++) {
          iz = iza;
          LFVolume* v1 = volume_i + i1;
          int M1 = v1->M;
          const SplineObject* spline_obj = spline_obj_M[M1];
          const bmgsspline* spline = \
            (const bmgsspline*)(&(spline_obj->spline));
          
          int nm1 = v1->nm;

          int M1p = MAX(M1, Mstart);
          int nm1p = MIN(M1 + nm1, Mstop) - M1p;
          if (nm1p <= 0)
            continue;


          double fdYdc_m[nm1];
          double rlYdfdr_m[nm1];
          double f, dfdr;
          int l = (nm1 - 1) / 2;
          //assert(2 * l + 1 == nm1);
          //assert(spline_obj->spline.l == l);
          const double* pos_c = pos_Wc[v1->W];

          int gm1 = 0;
          for (int G = Ga; G < Gb; G++, iz++) {
            double x = h_cv[0] * ix + h_cv[3] * iy + h_cv[6] * iz - pos_c[0];
            double y = h_cv[1] * ix + h_cv[4] * iy + h_cv[7] * iz - pos_c[1];
            double z = h_cv[2] * ix + h_cv[5] * iy + h_cv[8] * iz - pos_c[2];
            double vtdv = vt_G[G] * dv;

            double R_c[] = {x, y, z};
            
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double Rc_over_r = r > 1e-15 ? R_c[c] / r : 0.0;
            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            //assert (r <= spline->dr * spline->nbins);

            switch(c) {
            case 0:
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 1:
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdYdc_m);
              break;
            case 2:
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdYdc_m);
              break;
            }
            spherical_harmonics(l, dfdr * Rc_over_r, x, y, z, r2, rlYdfdr_m);
	    
            int m1start = M1 < Mstart ? nm1 - nm1p : 0;
            for (int m1 = 0; m1 < nm1p; m1++, gm1++) {
              work_gm[gm1] = vtdv * (fdYdc_m[m1 + m1start] 
				     + rlYdfdr_m[m1 + m1start]);
            }            
          } // end loop over G

          for (int i2 = 0; i2 < ni; i2++) {
            LFVolume* v2 = volume_i + i2;
            int M2 = v2->M;
            const double* A2_start_gm = v2->A_gm;
            const double* A2_gm;
            double complex* DVt_start_mm = DVt_MM + (M1p - Mstart) * nM + M2;
            double complex* DVt_mm;
            double complex work;
            int nm2 = v2->nm;
            double complex phase = conj(phase_i[i1]) * phase_i[i2];
            
            for (int g = 0; g < nG; g++) {
              A2_gm = A2_start_gm + g * nm2;
              for (int m1 = 0; m1 < nm1p; m1++) {
                work = work_gm[g * nm1p + m1] * phase;
                DVt_mm = DVt_start_mm + m1 * nM;
                for (int m2 = 0; m2 < nm2; m2++) {
                  DVt_mm[m2] += A2_gm[m2] * work;
                }
              }
            }
          } // i2 loop
        } // G loop
      } // i1 loop
      GRID_LOOP_STOP(lfc, k);
    } // c loop
  }
  Py_RETURN_NONE;
}

PyObject* derivative(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* a_xG_obj;
  PyArrayObject* c_xMv_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyObject* spline_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;
  int q;

  if (!PyArg_ParseTuple(args, "OOOOOOOi", &a_xG_obj, &c_xMv_obj,
                        &h_cv_obj, &n_c_obj,
                        &spline_M_obj, &beg_c_obj,
                        &pos_Wc_obj, &q))
    return NULL; 

  int nd = a_xG_obj->nd;
  npy_intp* dims = a_xG_obj->dimensions;
  int nx = PyArray_MultiplyList(dims, nd - 3);
  int nG = PyArray_MultiplyList(dims + nd - 3, 3);
  int nM = c_xMv_obj->dimensions[c_xMv_obj->nd - 2];

  const double* h_cv = (const double*)h_cv_obj->data;
  const long* n_c = (const long*)n_c_obj->data;
  const double (*pos_Wc)[3] = (const double (*)[3])pos_Wc_obj->data;

  long* beg_c = LONGP(beg_c_obj);

  if (!lfc->bloch_boundary_conditions) {
    const double* a_G = (const double*)a_xG_obj->data;
    double* c_Mv = (double*)c_xMv_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, -1) {
        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        for (int G = Ga; G < Gb; G++) {
          for (int i = 0; i < ni; i++) {
            LFVolume* vol = volume_i + i;
            int M = vol->M;
            double* c_mv = c_Mv + 3 * M;
            const bmgsspline* spline = (const bmgsspline*) \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
              
            int nm = vol->nm;
            int l = (nm - 1) / 2;
            double x = xG - pos_Wc[vol->W][0];
            double y = yG - pos_Wc[vol->W][1];
            double z = zG - pos_Wc[vol->W][2];
            double R_c[] = {x, y, z};
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double af;
            double dfdr;
            bmgs_get_value_and_derivative(spline, r, &af, &dfdr);
            af *= a_G[G] * lfc->dv;
            double afdrlYdx_m[nm];  // a * f * d(r^l * Y)/dx
            spherical_harmonics_derivative_x(l, af, x, y, z, r2, afdrlYdx_m);
            for (int m = 0; m < nm; m++)
              c_mv[3 * m] += afdrlYdx_m[m];
            spherical_harmonics_derivative_y(l, af, x, y, z, r2, afdrlYdx_m);
            for (int m = 0; m < nm; m++)
              c_mv[3 * m + 1] += afdrlYdx_m[m];
            spherical_harmonics_derivative_z(l, af, x, y, z, r2, afdrlYdx_m);
            for (int m = 0; m < nm; m++)
              c_mv[3 * m + 2] += afdrlYdx_m[m];
            if (r > 1e-15) {
              double arlm1Ydfdr_m[nm]; // a * r^(l-1) * Y * df/dr
              double arm1dfdr = a_G[G] / r * dfdr * lfc->dv;
              spherical_harmonics(l, arm1dfdr, x, y, z, r2, arlm1Ydfdr_m);
              for (int m = 0; m < nm; m++)
                for (int v = 0; v < 3; v++)
                  c_mv[m * 3 + v] += arlm1Ydfdr_m[m] * R_c[v];
            }
          }
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, -1);
      c_Mv += 3 * nM;
      a_G += nG;
    }
  }
  else {
    const complex double* a_G = (const complex double*)a_xG_obj->data;
    complex double* c_Mv = (complex double*)c_xMv_obj->data;
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, q) {
        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        for (int G = Ga; G < Gb; G++) {
          for (int i = 0; i < ni; i++) {
            LFVolume* vol = volume_i + i;
            int M = vol->M;
            complex double* c_mv = c_Mv + 3 * M;
            const bmgsspline* spline = (const bmgsspline*) \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
              
            int nm = vol->nm;
            int l = (nm - 1) / 2;
            double x = xG - pos_Wc[vol->W][0];
            double y = yG - pos_Wc[vol->W][1];
            double z = zG - pos_Wc[vol->W][2];
            double R_c[] = {x, y, z};
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double f;
            double dfdr;
            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            double fdrlYdx_m[nm];  // a * f * d(r^l * Y)/dx
            complex double ap = a_G[G] * phase_i[i] * lfc->dv;
            spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdrlYdx_m);
            for (int m = 0; m < nm; m++)
              c_mv[3 * m    ] += ap * fdrlYdx_m[m];
            spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdrlYdx_m);
            for (int m = 0; m < nm; m++)
              c_mv[3 * m + 1] += ap * fdrlYdx_m[m];
            spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdrlYdx_m);
            for (int m = 0; m < nm; m++)
              c_mv[3 * m + 2] += ap * fdrlYdx_m[m];
            if (r > 1e-15) {
              double rlm1Ydfdr_m[nm];  // r^(l-1) * Y * df/dr
              double rm1dfdr = dfdr / r;
              spherical_harmonics(l, rm1dfdr, x, y, z, r2, rlm1Ydfdr_m);
              for (int m = 0; m < nm; m++)
                for (int v = 0; v < 3; v++)
                  c_mv[m * 3 + v] += ap * rlm1Ydfdr_m[m] * R_c[v];
            }
          }
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, q);
      c_Mv += 3 * nM;
      a_G += nG;
    }
  }
  Py_RETURN_NONE;
}

PyObject* normalized_derivative(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* a_G_obj;
  PyArrayObject* c_Mv_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyObject* spline_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;

  if (!PyArg_ParseTuple(args, "OOOOOOO", &a_G_obj, &c_Mv_obj,
                        &h_cv_obj, &n_c_obj,
                        &spline_M_obj, &beg_c_obj,
                        &pos_Wc_obj))
    return NULL; 

  const double* h_cv = (const double*)h_cv_obj->data;
  const long* n_c = (const long*)n_c_obj->data;
  const double (*pos_Wc)[3] = (const double (*)[3])pos_Wc_obj->data;
  long* beg_c = LONGP(beg_c_obj);
  const double* a_G = (const double*)a_G_obj->data;
  double* c_Mv = (double*)c_Mv_obj->data;
  GRID_LOOP_START(lfc, -1) {
    int i2 = Ga % n_c[2] + beg_c[2];
    int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
    int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
    double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
    double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
    double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
    for (int G = Ga; G < Gb; G++) {
      for (int i = 0; i < ni; i++) {
        LFVolume* vol = volume_i + i;
        int M = vol->M;
        double* c_mv = c_Mv + 7 * M;
        const bmgsspline* spline = (const bmgsspline*)                  \
          &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
        
        int nm = vol->nm;
        int l = (nm - 1) / 2;
        double x = xG - pos_Wc[vol->W][0];
        double y = yG - pos_Wc[vol->W][1];
        double z = zG - pos_Wc[vol->W][2];
        double R_c[] = {x, y, z};
        double r2 = x * x + y * y + z * z;
        double r = sqrt(r2);
        double f;
        double dfdr;
        bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
        f *= lfc->dv;
        double a = a_G[G];
        if (l == 0)
          c_mv[6] += 0.28209479177387814 * a * f;
        double fdrlYdx_m[nm];  // f * d(r^l * Y)/dx
        spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdrlYdx_m);
        for (int m = 0; m < nm; m++) {
          c_mv[7 * m    ] += a * fdrlYdx_m[m];
          c_mv[7 * m + 3] += fdrlYdx_m[m];
        }
        spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdrlYdx_m);
        for (int m = 0; m < nm; m++) {
          c_mv[7 * m + 1] += a * fdrlYdx_m[m];
          c_mv[7 * m + 4] += fdrlYdx_m[m];
        }
        spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdrlYdx_m);
        for (int m = 0; m < nm; m++) {
          c_mv[7 * m + 2] += a * fdrlYdx_m[m];
          c_mv[7 * m + 5] += fdrlYdx_m[m];
        }
        if (r > 1e-15) {
          double rlm1Ydfdr_m[nm]; // r^(l-1) * Y * df/dr
          double rm1dfdr = dfdr * lfc->dv / r;
          spherical_harmonics(l, rm1dfdr, x, y, z, r2, rlm1Ydfdr_m);
          for (int m = 0; m < nm; m++)
            for (int v = 0; v < 3; v++) {
              c_mv[m * 7 + v] += a * rlm1Ydfdr_m[m] * R_c[v];
              c_mv[m * 7 + v + 3] += rlm1Ydfdr_m[m] * R_c[v];
            }
        }
      }
      xG += h_cv[6];
      yG += h_cv[7];
      zG += h_cv[8];
    }
  }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}

PyObject* ae_valence_density_correction(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* rho_MM_obj;
  PyArrayObject* n_G_obj;
  const PyArrayObject* a_W_obj;
  PyArrayObject* I_a_obj;

  if (!PyArg_ParseTuple(args, "OOOO", &rho_MM_obj, &n_G_obj,
                        &a_W_obj, &I_a_obj))
    return NULL; 
  
  double* n_G = (double*)n_G_obj->data;
  int* a_W = (int*)a_W_obj->data;
  double* I_a = (double*)I_a_obj->data;
  const double* rho_MM = (const double*)rho_MM_obj->data;

  int nM = rho_MM_obj->dimensions[0];

  GRID_LOOP_START(lfc, -1) {
    for (int i = 0; i < ni; i++) {
      LFVolume* v = volume_i + i;
      int M = v->M;
      int nm = v->nm;
      const double* rho_mm = rho_MM + M * nM + M;
      double Ia = 0.0;
      for (int g = 0; g < nG; g++) {
        double density = 0.0;
        for (int m2 = 0; m2 < nm; m2++)
          for (int m1 = 0; m1 < nm; m1++)
            density += (rho_mm[m2 + m1 * nM] *
                  v->A_gm[g * nm + m1] * v->A_gm[g * nm + m2]);
        n_G[Ga + g] += density;
        Ia += density;
      }
      I_a[a_W[v->W]] += Ia * lfc->dv;
    }
  }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}

PyObject* ae_core_density_correction(LFCObject *lfc, PyObject *args)
{
  double scale;
  PyArrayObject* n_G_obj;
  const PyArrayObject* a_W_obj;
  PyArrayObject* I_a_obj;

  if (!PyArg_ParseTuple(args, "dOOO", &scale, &n_G_obj,
                        &a_W_obj, &I_a_obj))
    return NULL; 
  
  double* n_G = (double*)n_G_obj->data;
  int* a_W = (int*)a_W_obj->data;
  double* I_a = (double*)I_a_obj->data;

  GRID_LOOP_START(lfc, -1) {
    for (int i = 0; i < ni; i++) {
      LFVolume* v = volume_i + i;
      double Ia = 0.0;
      for (int g = 0; g < nG; g++) {
        double density = scale * v->A_gm[g];
        n_G[Ga + g] += density;
        Ia += density;
      }
      I_a[a_W[v->W]] += Ia * lfc->dv;
    }
  }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}
