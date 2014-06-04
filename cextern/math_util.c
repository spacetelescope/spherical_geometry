#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "numpy/npy_3kcompat.h"

#include "numpy/npy_math.h"

/*
 *****************************************************************************
 **                            BASICS                                       **
 *****************************************************************************
 */

typedef npy_intp intp;

#define INIT_OUTER_LOOP_1       \
    intp dN = *dimensions++;    \
    intp N_;                    \
    intp s0 = *steps++;

#define INIT_OUTER_LOOP_2       \
    INIT_OUTER_LOOP_1           \
    intp s1 = *steps++;

#define INIT_OUTER_LOOP_3       \
    INIT_OUTER_LOOP_2           \
    intp s2 = *steps++;

#define INIT_OUTER_LOOP_4       \
    INIT_OUTER_LOOP_3           \
    intp s3 = *steps++;

#define INIT_OUTER_LOOP_5       \
    INIT_OUTER_LOOP_4           \
    intp s4 = *steps++;

#define BEGIN_OUTER_LOOP_3      \
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {

#define BEGIN_OUTER_LOOP_4      \
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2, args[3] += s3) {

#define BEGIN_OUTER_LOOP_5      \
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2, args[3] += s3, args[4] += s4) {

#define END_OUTER_LOOP  }

static inline void
load_point(const char *in, const intp s, double* out) {
    out[0] = (*(double *)in);
    in += s;
    out[1] = (*(double *)in);
    in += s;
    out[2] = (*(double *)in);
}

static inline void
save_point(const double* in, char *out, const intp s) {
    *(double *)out = in[0];
    out += s;
    *(double *)out = in[1];
    out += s;
    *(double *)out = in[2];
}

static inline void
cross(const double *A, const double *B, double *C) {
    C[0] = A[1]*B[2] - A[2]*B[1];
    C[1] = A[2]*B[0] - A[0]*B[2];
    C[2] = A[0]*B[1] - A[1]*B[0];
}

static inline void
normalize(double *A) {
    double l = sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);
    A[0] /= l;
    A[1] /= l;
    A[2] /= l;
}

static inline double
dot(const double *A, const double *B) {
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
}

static inline double
sign(const double A) {
    return (A == 0.0) ? 0.0 : ((A < 0.0) ? -1.0 : 1.0);
}

static inline int
equals(const double *A, const double *B) {
    return A[0] == B[0] && A[1] == B[1] && A[2] == B[2];
}

static inline void
mult(double *T, const double f) {
    T[0] *= f;
    T[1] *= f;
    T[2] *= f;
}

/*
 *****************************************************************************
 **                             UFUNC LOOPS                                 **
 *****************************************************************************
 */

/*///////////////////////////////////////////////////////////////////////////
  inner1d
*/

char *inner1d_signature = "(i),(i)->()";

static void
DOUBLE_inner1d(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    INIT_OUTER_LOOP_3
    intp di = dimensions[0];
    intp i;
    intp is1=steps[0], is2=steps[1];
    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];
        double sum = 0;
        for (i = 0; i < di; i++) {
            sum += (*(double *)ip1) * (*(double *)ip2);
            ip1 += is1;
            ip2 += is2;
        }
        *(double *)op = sum;
    END_OUTER_LOOP
}

static PyUFuncGenericFunction inner1d_functions[] = { DOUBLE_inner1d };
static void * inner1d_data[] = { (void *)NULL };
static char inner1d_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  cross
*/

char *cross_signature = "(i),(i)->(i)";

static void
DOUBLE_cross(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    double A[3];
    double B[3];
    double C[3];

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1], is3=steps[2];
    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_point(ip1, is1, A);
        load_point(ip2, is2, B);

        cross(A, B, C);

        save_point(C, op, is3);
    END_OUTER_LOOP
}

static PyUFuncGenericFunction cross_functions[] = { DOUBLE_cross };
static void * cross_data[] = { (void *)NULL };
static char cross_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  cross_and_norm
*/

char *cross_and_norm_signature = "(i),(i)->(i)";

/*
 *  This implements the function
 *        out[n] = sum_i { in1[n, i] * in2[n, i] }.
 */
static void
DOUBLE_cross_and_norm(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    double A[3];
    double B[3];
    double C[3];

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1], is3=steps[2];
    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_point(ip1, is1, A);
        load_point(ip2, is2, B);

        cross(A, B, C);
        normalize(C);

        save_point(C, op, is3);
    END_OUTER_LOOP
}

static PyUFuncGenericFunction cross_and_norm_functions[] = { DOUBLE_cross_and_norm };
static void * cross_and_norm_data[] = { (void *)NULL };
static char cross_and_norm_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  intersection
*/

char *intersection_signature = "(i),(i),(i),(i)->(i)";

/*
 * Finds the intersection of 2 great circle arcs AB and CD.
 */
static void
DOUBLE_intersection(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    double A[3];
    double B[3];
    double C[3];
    double D[3];

    double ABX[3];
    double CDX[3];
    double T[3];
    double tmp[3];

    double nans[3];

    double s;
    int match;

    nans[0] = nans[1] = nans[2] = NPY_NAN;

    INIT_OUTER_LOOP_5
    intp is1=steps[0], is2=steps[1], is3=steps[2], is4=steps[3], is5=steps[4];
    BEGIN_OUTER_LOOP_5
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *ip4=args[3], *op=args[4];

        load_point(ip1, is1, A);
        load_point(ip2, is2, B);
        load_point(ip3, is3, C);
        load_point(ip4, is4, D);

        match = !(equals(A, C) | equals(A, D) | equals(B, C) | equals(B, D));

        if (match) {
            cross(A, B, ABX);
            cross(C, D, CDX);
            cross(ABX, CDX, T);
            normalize(T);

            match = 0;
            cross(ABX, A, tmp);
            s = sign(dot(tmp, T));
            cross(B, ABX, tmp);
            if (s == sign(dot(tmp, T))) {
                cross(CDX, C, tmp);
                if (s == sign(dot(tmp, T))) {
                    cross(D, CDX, tmp);
                    if (s == sign(dot(tmp, T))) {
                        match = 1;
                    }
                }
            }
        }

        if (match) {
            mult(T, s);
            save_point(T, op, is5);
        } else {
            save_point(nans, op, is5);
        }
    END_OUTER_LOOP
}

static PyUFuncGenericFunction intersection_functions[] = { DOUBLE_intersection };
static void * intersection_data[] = { (void *)NULL };
static char intersection_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*
 *****************************************************************************
 **                            MODULE                                       **
 *****************************************************************************
 */

static void
addUfuncs(PyObject *dictionary) {
    PyObject *f;

    f = PyUFunc_FromFuncAndDataAndSignature(
        inner1d_functions, inner1d_data, inner1d_signatures, 2, 2, 1,
        PyUFunc_None, "inner1d",
        "inner on the last dimension and broadcast on the rest \n"      \
        "     \"(i),(i)->()\" \n",
        0, inner1d_signature);
    PyDict_SetItemString(dictionary, "inner1d", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        cross_functions, cross_data, cross_signatures, 2, 2, 1,
        PyUFunc_None, "cross",
        "cross product of 3-vectors only \n" \
        "     \"(i),(i)->(i)\" \n",
        0, cross_signature);
    PyDict_SetItemString(dictionary, "cross", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        cross_and_norm_functions, cross_and_norm_data, cross_and_norm_signatures, 2, 2, 1,
        PyUFunc_None, "cross_and_norm",
        "cross_and_norm product of 3-vectors only \n" \
        "     \"(i),(i)->(i)\" \n",
        0, cross_and_norm_signature);
    PyDict_SetItemString(dictionary, "cross_and_norm", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        intersection_functions, intersection_data, intersection_signatures, 2, 4, 1,
        PyUFunc_None, "intersection",
        "intersection product of 3-vectors only \n" \
        "     \"(i),(i),(i),(i)->(i)\" \n",
        0, intersection_signature);
    PyDict_SetItemString(dictionary, "intersection", f);
    Py_DECREF(f);
}

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "math_util",
        NULL,
        -1,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyObject *PyInit_math_util(void)
#else
#define RETVAL
PyMODINIT_FUNC
initmath_util(void)
#endif
{
    PyObject *m;
    PyObject *d;

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("math_util", NULL);
#endif
    if (m == NULL)
        return RETVAL;

    import_array();
    import_ufunc();

    d = PyModule_GetDict(m);

    /* Load the ufunc operators into the module's namespace */
    addUfuncs(d);

    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load umath_tests module.");
    }

    return RETVAL;
}
