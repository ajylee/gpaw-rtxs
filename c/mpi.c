/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2005-2009  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#ifdef PARALLEL
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <mpi.h>
#include "extensions.h"
#include <structmember.h>
#include "mympi.h"

// Check that array is well-behaved and contains data that can be sent.
#define CHK_ARRAY(a) if ((a) == NULL || !PyArray_Check(a)		\
			 || !PyArray_ISCARRAY(a) || !PyArray_ISNUMBER(a)) { \
    PyErr_SetString(PyExc_TypeError,					\
		    "Not a proper NumPy array for MPI communication."); \
    return NULL; }

// Check that two arrays have the same type, and the size of the
// second is a given multiple of the size of the first
#define CHK_ARRAYS(a,b,n)						\
  if ((PyArray_TYPE(a) != PyArray_TYPE(b))				\
      || (PyArray_SIZE(b) != PyArray_SIZE(a) * n)) {			\
    PyErr_SetString(PyExc_ValueError,					\
		    "Incompatible array types or sizes.");		\
      return NULL; }

// Check that a processor number is valid
#define CHK_PROC(n) if (n < 0 || n >= self->size) {\
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");	\
    return NULL; }

// Check that a processor number is valid or is -1
#define CHK_PROC_DEF(n) if (n < -1 || n >= self->size) {\
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");	\
    return NULL; }

// Check that a processor number is valid and is not this processor
#define CHK_OTHER_PROC(n) if (n < 0 || n >= self->size || n == self->rank) { \
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");	\
    return NULL; }

// Poor mans MPI request object, so we can store a reference to the buffer,
// preventing its early deallocation.  This should be replaced by a real
// (opaque) python object, so we can detect waiting multiple times on the same
// object.
typedef struct {
  MPI_Request rq;
  PyObject *buffer;
} mpi_request;

static void mpi_dealloc(MPIObject *obj)
{
  if (obj->comm != MPI_COMM_WORLD)
    MPI_Comm_free(&(obj->comm));
  Py_XDECREF(obj->parent);
  free(obj->members);
  PyObject_DEL(obj);
}

static PyObject * mpi_sendreceive(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  PyObject* b;
  int dest, src;
  int sendtag = 123;
  int recvtag = 123;
  int ret; 
  static char *kwlist[] = {"a", "dest", "sendtag", "b", "src", "recvtag", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiOi|ii:sendreceive", kwlist,
				   &a, &dest, &b, &src, &sendtag, &recvtag))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(dest);
  CHK_ARRAY(b);
  CHK_OTHER_PROC(src);
  int nsend = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    nsend *= PyArray_DIM(a,d);
  int nrecv = PyArray_DESCR(b)->elsize;
  for (int d = 0; d < PyArray_NDIM(b); d++)
    nrecv *= PyArray_DIM(b,d);
  ret = MPI_Sendrecv(PyArray_BYTES(a), nsend, MPI_BYTE, dest, sendtag, 
		     PyArray_BYTES(b), nrecv, MPI_BYTE, src, recvtag, 
		     self->comm, MPI_STATUS_IGNORE);
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Sendrecv error occured.");
	  return NULL;
	}
#endif
      Py_RETURN_NONE;
    }


static PyObject * mpi_receive(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int src;
  int tag = 123;
  int block = 1;
  int ret;
  static char *kwlist[] = {"a", "src", "tag", "block", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii:receive", kwlist,
				   &a, &src, &tag, &block))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(src);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a, d);
  if (block)
    {
      ret = MPI_Recv(PyArray_BYTES(a), n, MPI_BYTE, src, tag, self->comm,
	       MPI_STATUS_IGNORE);
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Recv error occured.");
	  return NULL;
	}
#endif
      Py_RETURN_NONE;
    }
  else
    {
      mpi_request req;
      req.buffer = a;
      Py_INCREF(req.buffer);
      ret = MPI_Irecv(PyArray_BYTES(a), n, MPI_BYTE, src, tag, self->comm, &(req.rq));
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Irecv error occured.");
	  return NULL;
	}
#endif
      return Py_BuildValue("s#", &req, sizeof(req));
    }
}

static PyObject * mpi_send(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int dest;
  int tag = 123;
  int block = 1;
  int ret; 
  static char *kwlist[] = {"a", "dest", "tag", "block", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii:send", kwlist,
				   &a, &dest, &tag, &block))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(dest);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  if (block)
    {
      ret = MPI_Send(PyArray_BYTES(a), n, MPI_BYTE, dest, tag, self->comm);
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Send error occured.");
	  return NULL;
	}
#endif
      Py_RETURN_NONE;
    }
  else
    {
      mpi_request req;
      req.buffer = a;
      Py_INCREF(a);
      ret = MPI_Isend(PyArray_BYTES(a), n, MPI_BYTE, dest, tag, self->comm,
		&(req.rq));
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Isend error occured.");
	  return NULL;
	}
#endif
      return Py_BuildValue("s#", &req, sizeof(req));
    }
}

static PyObject * mpi_ssend(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int dest;
  int tag = 123;
  static char *kwlist[] = {"a", "dest", "tag", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i:send", kwlist,
				   &a, &dest, &tag))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(dest);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  MPI_Ssend(PyArray_BYTES(a), n, MPI_BYTE, dest, tag, self->comm);
  Py_RETURN_NONE;
}


static PyObject * mpi_name(MPIObject *self, PyObject *noargs)
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlen;
  MPI_Get_processor_name(name, &resultlen);
  return Py_BuildValue("s#", name, resultlen);
}

static PyObject * mpi_abort(MPIObject *self, PyObject *args)
{
  int errcode;
  if (!PyArg_ParseTuple(args, "i:abort", &errcode))
    return NULL;
  MPI_Abort(self->comm, errcode);
  Py_RETURN_NONE;
}

static PyObject * mpi_barrier(MPIObject *self)
{
  MPI_Barrier(self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_test(MPIObject *self, PyObject *args)
{
  mpi_request* s;
  int n;
  int ret;
  int flag;
  if (!PyArg_ParseTuple(args, "s#:test", &s, &n))
    return NULL;
  if (n != sizeof(mpi_request))
    {
      PyErr_SetString(PyExc_TypeError, "Invalid MPI request object.");
      return NULL;
    }
  ret = MPI_Test(&(s->rq), &flag, MPI_STATUS_IGNORE); // Can this change the Python string?
#ifdef GPAW_MPI_DEBUG
  if (ret != MPI_SUCCESS)
    {
      PyErr_SetString(PyExc_RuntimeError, "MPI_Test error occured.");
      return NULL;
    }
#endif
  // Note that MPI_Test does not modify the request regardless of flag outcome.
  return Py_BuildValue("i", flag);
}

static PyObject * mpi_testall(MPIObject *self, PyObject *requests)
{
  int n;   // Number of requests
  int ret;
  MPI_Request *rqs = NULL;
  PyObject **bufs = NULL;
  int flag = 0;
  if (!PySequence_Check(requests))
    {
      PyErr_SetString(PyExc_TypeError, "mpi.testall: argument must be a sequence");
      return NULL;
    }
  // Extract the request objects 
  n = PySequence_Size(requests);
  assert(n >= 0);  // This cannot fail.
  rqs = GPAW_MALLOC(MPI_Request, n);
  bufs = GPAW_MALLOC(PyObject*, n);
  for (int i = 0; i < n; i++)
    {
      PyObject *o = PySequence_GetItem(requests, i);
      if (o == NULL)
	return NULL;
      if (!PyString_CheckExact(o) || (PyString_Size(o) != sizeof(mpi_request)))
	{
	  Py_DECREF(o);
	  free(rqs);
	  free(bufs);
	  PyErr_SetString(PyExc_TypeError, "mpi.testall: argument must be a sequence of MPI requests");
	  return NULL;
	}
      mpi_request *s = (mpi_request *) PyString_AS_STRING(o);
      //memcpy(rqs[i], &(s->rq), sizeof(MPI_Request));
      rqs[i] = s->rq;
      bufs[i] = s->buffer;
      Py_DECREF(o);
    }
  // Do the actual test.
  ret = MPI_Testall(n, rqs, &flag, MPI_STATUSES_IGNORE);
#ifdef GPAW_MPI_DEBUG
  if (ret != MPI_SUCCESS)
    {
      // We do not dare to release the buffers now!
      PyErr_SetString(PyExc_RuntimeError, "MPI_Testall error occured.");
      return NULL;
    }
#endif
  // Unlike MPI_Test, if flag outcome is non-zero, MPI_Testall will deallocate
  // all requests which were allocated by nonblocking communication calls, so
  // we must free these buffers. Otherwise, none of the requests are modified.
  if (flag != 0)
    {
      // Release the buffers used by the MPI communication
      for (int i = 0; i < n; i++)
        Py_DECREF(bufs[i]);
    }
  // Release internal data and return.
  free(rqs);
  free(bufs);
  return Py_BuildValue("i", flag);
}

static PyObject * mpi_wait(MPIObject *self, PyObject *args)
{
  mpi_request* s;
  int n;
  int ret;
  if (!PyArg_ParseTuple(args, "s#:wait", &s, &n))
    return NULL;
  if (n != sizeof(mpi_request))
    {
      PyErr_SetString(PyExc_TypeError, "Invalid MPI request object.");
      return NULL;
    }
  ret = MPI_Wait(&(s->rq), MPI_STATUS_IGNORE); // Can this change the Python string?
  Py_DECREF(s->buffer);
#ifdef GPAW_MPI_DEBUG
  if (ret != MPI_SUCCESS)
    {
      PyErr_SetString(PyExc_RuntimeError, "MPI_Wait error occured.");
      return NULL;
    }
#endif
  Py_RETURN_NONE;
}

static PyObject * mpi_waitall(MPIObject *self, PyObject *requests)
{
  int n;   // Number of requests
  int ret;
  MPI_Request *rqs = NULL;
  PyObject **bufs = NULL;
  if (!PySequence_Check(requests))
    {
      PyErr_SetString(PyExc_TypeError, "mpi.waitall: argument must be a sequence");
      return NULL;
    }
  // Extract the request objects 
  n = PySequence_Size(requests);
  assert(n >= 0);  // This cannot fail.
  rqs = GPAW_MALLOC(MPI_Request, n);
  bufs = GPAW_MALLOC(PyObject*, n);
  for (int i = 0; i < n; i++)
    {
      PyObject *o = PySequence_GetItem(requests, i);
      if (o == NULL)
	return NULL;
      if (!PyString_CheckExact(o) || (PyString_Size(o) != sizeof(mpi_request)))
	{
	  Py_DECREF(o);
	  free(rqs);
	  free(bufs);
	  PyErr_SetString(PyExc_TypeError, "mpi.waitall: argument must be a sequence of MPI requests");
	  return NULL;
	}
      mpi_request *s = (mpi_request *) PyString_AS_STRING(o);
      //memcpy(rqs[i], &(s->rq), sizeof(MPI_Request));
      rqs[i] = s->rq;
      bufs[i] = s->buffer;
      Py_DECREF(o);
    }
  // Do the actual wait.
  ret = MPI_Waitall(n, rqs, MPI_STATUSES_IGNORE);
#ifdef GPAW_MPI_DEBUG
  if (ret != MPI_SUCCESS)
    {
      // We do not dare to release the buffers now!
      PyErr_SetString(PyExc_RuntimeError, "MPI_Waitall error occured.");
      return NULL;
    }
#endif
  // Release the buffers used by the MPI communication
  for (int i = 0; i < n; i++)
    Py_DECREF(bufs[i]);
  // Release internal data and return.
  free(rqs);
  free(bufs);
  Py_RETURN_NONE;
}
 

static MPI_Datatype get_mpi_datatype(PyObject *a)
{
  int n = PyArray_DESCR(a)->elsize;
  if (PyArray_ISCOMPLEX(a))
    n = n/2;

  switch(PyArray_TYPE(a))
    {
      // Floating point numbers including complex numbers
    case NPY_DOUBLE:
    case NPY_CDOUBLE:
      assert(sizeof(double) == n);
      return MPI_DOUBLE;
    case NPY_FLOAT:
    case NPY_CFLOAT:
      assert(sizeof(float) == n);
      return MPI_FLOAT;
    case NPY_LONGDOUBLE:
    case NPY_CLONGDOUBLE:
       assert(sizeof(long double) == n);
      return MPI_LONG_DOUBLE;
      // Signed integer types
    case NPY_BYTE:
      assert(sizeof(char) == n);
      return MPI_CHAR;
    case NPY_SHORT:
      assert(sizeof(short) == n);
      return MPI_SHORT;
    case NPY_INT:
      assert(sizeof(int) == n);
      return MPI_INT;
    case NPY_LONG:
      assert(sizeof(long) == n);
      return MPI_LONG;
      // Unsigned integer types
    case NPY_BOOL:
    case NPY_UBYTE:
      assert(sizeof(unsigned char) == n);
      return MPI_UNSIGNED_CHAR;
    case NPY_USHORT:
      assert(sizeof(unsigned short) == n);
      return MPI_UNSIGNED_SHORT;
    case NPY_UINT:
      assert(sizeof(unsigned) == n);
      return MPI_UNSIGNED;
    case NPY_ULONG:
      assert(sizeof(unsigned long) == n);
      return MPI_UNSIGNED_LONG;
    }
  // If we reach this point none of the cases worked out.
  PyErr_SetString(PyExc_ValueError, "Cannot communicate data of this type.");
  return 0;
}

static PyObject * mpi_reduce(MPIObject *self, PyObject *args, PyObject *kwargs,
			     MPI_Op operation, int allowcomplex)
{
#ifdef GPAW_MPI_DEBUG
  MPI_Barrier(self->comm);
#endif
  PyObject* obj;
  int root = -1;
  int ret;
  static char *kwlist[] = {"a", "root", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:reduce", kwlist,
				   &obj, &root))
    return NULL;
  CHK_PROC_DEF(root);
  if (PyFloat_Check(obj))
    {
      double din = PyFloat_AS_DOUBLE(obj);
      double dout;
      if (root == -1)
        ret = MPI_Allreduce(&din, &dout, 1, MPI_DOUBLE, operation, self->comm);
      else
        ret = MPI_Reduce(&din, &dout, 1, MPI_DOUBLE, operation, root, 
                         self->comm);
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
        {
	  PyErr_SetString(PyExc_RuntimeError, 
                          "MPI_Allreduce/reduce DOUBLE error.");
	  return NULL;
        }	
#endif
      return PyFloat_FromDouble(dout);
    }
  if (PyInt_Check(obj))
    {
      long din = PyInt_AS_LONG(obj);
      long dout;
      if (root == -1)
        ret = MPI_Allreduce(&din, &dout, 1, MPI_LONG, operation, self->comm);
      else
        ret = MPI_Reduce(&din, &dout, 1, MPI_LONG, operation, root, 
                         self->comm);
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
        {
	  PyErr_SetString(PyExc_RuntimeError, 
                          "MPI_Allreduce/reduce LONG error.");
	  return NULL;
        }	
#endif
      return PyInt_FromLong(dout);
    }
  else if (PyComplex_Check(obj) && allowcomplex)
    {
      double din[2];
      double dout[2];
      din[0] = PyComplex_RealAsDouble(obj);
      din[1] = PyComplex_ImagAsDouble(obj);
      if (root == -1)
        ret = MPI_Allreduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, self->comm);
      else
        ret = MPI_Reduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, root, 
                         self->comm);
#ifdef GPAW_MPI_DEBUG
      if (ret != MPI_SUCCESS)
        {
	  PyErr_SetString(PyExc_RuntimeError, 
                          "MPI_Allreduce/reduce COMPLEX error.");
	  return NULL;
        }
#endif
      return PyComplex_FromDoubles(dout[0], dout[1]);
    }
  else if (PyComplex_Check(obj))
    {
      PyErr_SetString(PyExc_ValueError,
		      "Operation not allowed on complex numbers");
      return NULL;
    }   
  else   // It should be an array
    {
      int n;
      int elemsize;
      MPI_Datatype datatype;
      CHK_ARRAY(obj);
      datatype = get_mpi_datatype(obj);
      if (datatype == 0)
	return NULL;
      n = PyArray_SIZE(obj);
      elemsize = PyArray_DESCR(obj)->elsize;
      if (PyArray_ISCOMPLEX(obj))
	{
	  if (allowcomplex)
	    {
	      n *= 2;
	      elemsize /= 2;
	    }
	  else
	    {
	      PyErr_SetString(PyExc_ValueError,
			      "Operation not allowed on complex numbers");
	      return NULL;
	    }
	}
      if (root == -1)
        {
#ifdef GPAW_MPI2
	  ret = MPI_Allreduce(MPI_IN_PLACE, PyArray_BYTES(obj), n, datatype,
                        operation, self->comm);
#else
          char* b = GPAW_MALLOC(char, n * elemsize);
          ret = MPI_Allreduce(PyArray_BYTES(obj), b, n, datatype, operation,
			self->comm);
	  assert(PyArray_NBYTES(obj) == n * elemsize);
          memcpy(PyArray_BYTES(obj), b, n * elemsize);
          free(b);
#endif
#ifdef GPAW_MPI_DEBUG
          if (ret != MPI_SUCCESS)
	    {
	      PyErr_SetString(PyExc_RuntimeError, 
                              "MPI_Allreduce Array error.");
	      return NULL;
	    }	
#endif
        }
      else
        {
          int rank;
          MPI_Comm_rank(self->comm, &rank);
          char* b = 0;
          if (rank == root)
            {
#ifdef GPAW_MPI2 
              ret = MPI_Reduce(MPI_IN_PLACE, PyArray_BYTES(obj), n, 
                               datatype, operation, root, self->comm);
#else
              b = GPAW_MALLOC(char, n * elemsize);
              ret = MPI_Reduce(PyArray_BYTES(obj), b, n, datatype, 
                               operation, root, self->comm);
	      assert(PyArray_NBYTES(obj) == n * elemsize);
              memcpy(PyArray_BYTES(obj), b, n * elemsize);
              free(b);               
#endif
            }
          else
            {
              ret = MPI_Reduce(PyArray_BYTES(obj), b, n, datatype, 
                               operation, root, self->comm);
            }
#ifdef GPAW_MPI_DEBUG
          if (ret != MPI_SUCCESS)
            {
	      PyErr_SetString(PyExc_RuntimeError, 
                              "MPI_Reduce Array error occured.");
              return NULL;
	    }
#endif
        }
      Py_RETURN_NONE;
    }
}

static PyObject * mpi_sum(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args, kwargs, MPI_SUM, 1);
}

static PyObject * mpi_product(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  // No complex numbers as that would give separate products of
  // real and imaginary parts.
  return mpi_reduce(self, args, kwargs,  MPI_PROD, 0); 
}

static PyObject * mpi_max(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args,  kwargs, MPI_MAX, 0);
}

static PyObject * mpi_min(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args,  kwargs, MPI_MIN, 0);
}

static PyObject * mpi_scatter(MPIObject *self, PyObject *args)
{
  PyObject* sendobj;
  PyObject* recvobj;
  int root;
  if (!PyArg_ParseTuple(args, "OOi:scatter", &sendobj, &recvobj, &root))
    return NULL;
  CHK_ARRAY(recvobj);
  CHK_PROC(root);
  char* source = 0;
  if (self->rank == root) {
    CHK_ARRAY(sendobj);
    CHK_ARRAYS(recvobj, sendobj, self->size); // size(send) = size(recv)*Ncpu
    source = PyArray_BYTES(sendobj);
  }
  int n = PyArray_DESCR(recvobj)->elsize;
  for (int d = 0; d < PyArray_NDIM(recvobj); d++)
    n *= PyArray_DIM(recvobj,d);
  MPI_Scatter(source, n, MPI_BYTE, PyArray_BYTES(recvobj),
	      n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}



static PyObject * mpi_allgather(MPIObject *self, PyObject *args)
{
  PyArrayObject* a;
  PyArrayObject* b;
  if (!PyArg_ParseTuple(args, "OO:allgather", &a, &b))
    return NULL;
  CHK_ARRAY(a);
  CHK_ARRAY(b);
  CHK_ARRAYS(a, b, self->size);
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  // What about endianness???? 
  MPI_Allgather(PyArray_BYTES(a), n, MPI_BYTE, PyArray_BYTES(b), n,
		MPI_BYTE, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_gather(MPIObject *self, PyObject *args)
{
  PyObject* a;
  int root;
  PyObject* b = 0;
  if (!PyArg_ParseTuple(args, "Oi|O", &a, &root, &b))
    return NULL;
  CHK_ARRAY(a);
  CHK_PROC(root);
  if (root == self->rank)
    {
      CHK_ARRAY(b);
      CHK_ARRAYS(a, b, self->size);
    }
  else if (b != Py_None && b != NULL)
    {
      fprintf(stderr, "******** Root=%d\n", root);
      PyErr_SetString(PyExc_ValueError,	 
		      "mpi_gather: b array should not be given on non-root processors.");
      return NULL;
    }
  int n = PyArray_DESCR(a)->elsize;
  for (int d = 0; d < PyArray_NDIM(a); d++)
    n *= PyArray_DIM(a,d);
  if (root != self->rank)
    MPI_Gather(PyArray_BYTES(a), n, MPI_BYTE, 0, n, MPI_BYTE, root, self->comm);
  else
    MPI_Gather(PyArray_BYTES(a), n, MPI_BYTE, PyArray_BYTES(b), n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_broadcast(MPIObject *self, PyObject *args)
{
#ifdef GPAW_MPI_DEBUG
  MPI_Barrier(self->comm);
#endif
  PyObject* buf;
  int root;
  if (!PyArg_ParseTuple(args, "Oi:broadcast", &buf, &root))
    return NULL;
  CHK_ARRAY(buf);
  CHK_PROC(root);
  int n = PyArray_DESCR(buf)->elsize;
  for (int d = 0; d < PyArray_NDIM(buf); d++)
    n *= PyArray_DIM(buf,d);
  MPI_Bcast(PyArray_BYTES(buf), n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}

static PyObject * get_members(MPIObject *self, PyObject *args)
{
  PyArrayObject *ranks;
  npy_intp ranks_dims[1] = {self->size};
  ranks = (PyArrayObject *) PyArray_SimpleNew(1, ranks_dims, NPY_INT);
  if (ranks == NULL)
    return NULL;
  memcpy(INTP(ranks), self->members, self->size*sizeof(int));
  PyObject* values = Py_BuildValue("O", ranks);
  Py_DECREF(ranks);
  return values;
}

// See the documentation for corresponding function in debug wrapper
// for the purpose of this function (gpaw/mpi/__init__.py)
static PyObject * get_c_object(MPIObject *self, PyObject *args)
{
  return Py_BuildValue("O", self);
}

// Forward declaration of MPI_Communicator because it needs MPIType
// that needs MPI_getattr that needs MPI_Methods that need
// MPI_Communicator that need ...
static PyObject * MPICommunicator(MPIObject *self, PyObject *args);

static PyMethodDef mpi_methods[] = {
    {"sendreceive",          (PyCFunction)mpi_sendreceive,
     METH_VARARGS|METH_KEYWORDS,
     "sendreceive(a, dest, desttag=123, b, src, srctag=123) sends an array to dest and receives an array a from src."},
    {"receive",          (PyCFunction)mpi_receive,
     METH_VARARGS|METH_KEYWORDS,
     "receive(a, src, tag=123, block=1) receives array a from src."},
    {"send",             (PyCFunction)mpi_send,
     METH_VARARGS|METH_KEYWORDS,
     "send(a, dest, tag=123, block=1) sends array a to dest."},
    {"ssend",             (PyCFunction)mpi_ssend,
     METH_VARARGS|METH_KEYWORDS,
     "ssend(a, dest, tag=123) synchronously sends array a to dest."},
    {"abort",            (PyCFunction)mpi_abort,        METH_VARARGS,
     "abort(errcode) aborts all MPI tasks."},
    {"name",             (PyCFunction)mpi_name,         METH_NOARGS,
     "name() returns the name of the processor node."},
    {"barrier",          (PyCFunction)mpi_barrier,      METH_VARARGS,
     "barrier() synchronizes all MPI tasks"},
    {"test",             (PyCFunction)mpi_test,         METH_VARARGS,
     "test(request) tests if a nonblocking communication is complete."},
    {"testall",          (PyCFunction)mpi_testall,      METH_O,
     "testall(list_of_rqs) tests if multiple nonblocking communications are complete."},
    {"wait",             (PyCFunction)mpi_wait,         METH_VARARGS,
     "wait(request) waits for a nonblocking communication to complete."},
    {"waitall",          (PyCFunction)mpi_waitall,      METH_O,
     "waitall(list_of_rqs) waits for multiple nonblocking communications to complete."},
    {"sum",              (PyCFunction)mpi_sum,
     METH_VARARGS|METH_KEYWORDS,
     "sum(a, root=-1) sums arrays, result on all tasks unless root is given."},
    {"product",          (PyCFunction)mpi_product,
     METH_VARARGS|METH_KEYWORDS,
     "product(a, root=-1) multiplies arrays, result on all tasks unless root is given."},
    {"max",              (PyCFunction)mpi_max,
     METH_VARARGS|METH_KEYWORDS,
     "max(a, root=-1) maximum of arrays, result on all tasks unless root is given."},
    {"min",              (PyCFunction)mpi_min,
     METH_VARARGS|METH_KEYWORDS,
     "min(a, root=-1) minimum of arrays, result on all tasks unless root is given."},
    {"scatter",          (PyCFunction)mpi_scatter,      METH_VARARGS,
     "scatter(src, target, root) distributes array from root task."},
    {"gather",           (PyCFunction)mpi_gather,       METH_VARARGS,
     "gather(src, root, target=None) gathers data from all tasks on root task."},
    {"all_gather",       (PyCFunction)mpi_allgather,    METH_VARARGS,
     "all_gather(src, target) gathers data from all tasks on all tasks."},
    {"broadcast",        (PyCFunction)mpi_broadcast,    METH_VARARGS,
     "broadcast(buffer, root) Broadcast data in-place from root task."},
    {"get_members",      (PyCFunction)get_members,      METH_VARARGS, 0},
    {"get_c_object",     (PyCFunction)get_c_object,     METH_VARARGS, 0},
    {"new_communicator", (PyCFunction)MPICommunicator,  METH_VARARGS,
     "new_communicator(ranks) creates a new communicator."},
    {0, 0, 0, 0}
};

static PyMemberDef mpi_members[] = {
  {"size", T_INT, offsetof(MPIObject, size), 0, "Number of processors"},
  {"rank", T_INT, offsetof(MPIObject, rank), 0, "Number of this processor"},
  {"parent", T_OBJECT_EX, offsetof(MPIObject, parent), 0, "Parent communicator"},
  {0, 0, 0, 0, 0}  /* Sentinel */
};

// __new__
static PyObject *NewMPIObject(PyTypeObject* type, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {NULL};
  MPIObject* self;
  
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return NULL;

  self = (MPIObject *) type->tp_alloc(type, 0);
  if (self == NULL)
    return NULL;
  
  MPI_Comm_size(MPI_COMM_WORLD, &(self->size));
  MPI_Comm_rank(MPI_COMM_WORLD, &(self->rank));
  self->comm = MPI_COMM_WORLD;
  Py_INCREF(Py_None);
  self->parent = Py_None;
  self->members = (int*) malloc(self->size*sizeof(int));
  if (self->members == NULL)
    return NULL;
  for (int i=0; i<self->size; i++)
    self->members[i] = i;

  return (PyObject *) self;
}

// __init__ does nothing.
static int InitMPIObject(MPIObject* self, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {NULL};
  
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return -1;
  
  return 0;
}


PyTypeObject MPIType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                         /*ob_size*/
  "MPI",                     /*tp_name*/
  sizeof(MPIObject),         /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)mpi_dealloc,   /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash*/
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "MPI object",              /*tp_doc*/
  0,                         /*tp_traverse*/
  0,                         /*tp_clear*/
  0,                         /*tp_richcompare*/
  0,                         /*tp_weaklistoffset*/
  0,                         /*tp_iter*/
  0,                         /*tp_iternext*/
  mpi_methods,               /*tp_methods*/
  mpi_members,               /*tp_members*/
  0,                         /*tp_getset*/
  0,                         /*tp_base*/
  0,                         /*tp_dict*/
  0,                         /*tp_descr_get*/
  0,                         /*tp_descr_set*/
  0,                         /*tp_dictoffset*/
  (initproc)InitMPIObject,   /*tp_init*/
  0,                         /*tp_alloc*/
  NewMPIObject,              /*tp_new*/
};


static PyObject * MPICommunicator(MPIObject *self, PyObject *args)
{
  PyObject* orig_ranks;
  if (!PyArg_ParseTuple(args, "O", &orig_ranks))
    return NULL;
  // NB: int32 is NPY_LONG on 32-bit Linux and NPY_INT on 64-bit Linux!
  // First convert to NumPy array of NPY_LONG, then cast to NPY_INT, to 
  // allow both 32 and 64 bit integers in the argument (except 64 on 32).
  PyObject *ranks = PyArray_ContiguousFromAny(orig_ranks, NPY_LONG, 1, 1);
  if (ranks == NULL)
    return NULL;
  PyObject *iranks;
  int n = PyArray_DIM(ranks, 0);
  iranks = PyArray_Cast((PyArrayObject*) ranks, NPY_INT);
  Py_DECREF(ranks);
  if (iranks == NULL)
    return NULL;
  // Check that all ranks make sense
  for (int i = 0; i < n; i++)
    {
      int *x = PyArray_GETPTR1(iranks, i);
      if (*x < 0 || *x >= self->size)
	{
	  Py_DECREF(iranks);
	  PyErr_SetString(PyExc_ValueError, "invalid rank");
	  return NULL;
	}
      for (int j = 0; j < i; j++)
	{
	  int *y = PyArray_GETPTR1(iranks, j);
	  if (*y == *x)
	    {
	      Py_DECREF(iranks);
	      PyErr_SetString(PyExc_ValueError, "duplicate rank");
	      return NULL;
	    }
	}
    }
  MPI_Group group;
  MPI_Comm_group(self->comm, &group);
  MPI_Group newgroup;
  MPI_Group_incl(group, n, (int *) PyArray_BYTES(iranks), &newgroup);
  MPI_Comm comm;
  MPI_Comm_create(self->comm, newgroup, &comm); // has a memory leak!
#ifdef GPAW_MPI_DEBUG
  // Default Errhandler is MPI_ERRORS_ARE_FATAL
  MPI_Errhandler_set(comm, MPI_ERRORS_RETURN); 
#endif 
  MPI_Group_free(&newgroup);
  MPI_Group_free(&group);
  if (comm == MPI_COMM_NULL)
    {
      Py_DECREF(iranks);
      Py_RETURN_NONE;
    }
  else
    {
      MPIObject *obj = PyObject_NEW(MPIObject, &MPIType);
      if (obj == NULL)
        return NULL;
      MPI_Comm_size(comm, &(obj->size));
      MPI_Comm_rank(comm, &(obj->rank));
      obj->comm = comm;
      if (obj->parent == Py_None)
        Py_DECREF(obj->parent);
      obj->members = (int*) malloc(obj->size*sizeof(int));
      if (obj->members == NULL)
        return NULL;
      memcpy(obj->members, (int *) PyArray_BYTES(iranks), obj->size*sizeof(int));
      Py_DECREF(iranks);

      // Make sure that MPI_COMM_WORLD is kept alive till the end (we
      // don't want MPI_Finalize to be called before MPI_Comm_free):
      Py_INCREF(self);
      obj->parent = (PyObject*)self;
      return (PyObject*)obj;
    }
}
#endif //PARALLEL
