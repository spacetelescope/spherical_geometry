#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "numpy/npy_3kcompat.h"

#include "numpy/npy_math.h"

/*
  The intersects, length and intersects_point calculations use "long
  doubles" internally.  Emperical testing shows that this level of
  precision is required for finding all intersections in typical HST
  images that are rotated only slightly from one another.

  Unfortunately, "long double" is not a standard: On x86_64 Linux and
  Mac, this is an 80-bit floating point representation, though
  reportedly it is equivalent to double on Windows.  If
  Windows-specific or non x86_64 problems present themselves, we may
  want to use a software floating point library or some other method
  for this support.
*/

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

#define BEGIN_OUTER_LOOP_2      \
    for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1) {

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
load_pointl(const char *in, const intp s, long double* out) {
    out[0] = (long double)(*(double *)in);
    in += s;
    out[1] = (long double)(*(double *)in);
    in += s;
    out[2] = (long double)(*(double *)in);
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
save_pointl(const long double* in, char *out, const intp s) {
    *(double *)out = (double)in[0];
    out += s;
    *(double *)out = (double)in[1];
    out += s;
    *(double *)out = (double)in[2];
}

static inline void
crossl(const long double *A, const long double *B, long double *C) {
    C[0] = A[1]*B[2] - A[2]*B[1];
    C[1] = A[2]*B[0] - A[0]*B[2];
    C[2] = A[0]*B[1] - A[1]*B[0];
}

static inline void
normalize_output(long double *A, double *B) {
    double l = A[0]*A[0] + A[1]*A[1] + A[2]*A[2];
    if (l != 1.0) {
        l = sqrt(l);
        B[0] = A[0] / l;
        B[1] = A[1] / l;
        B[2] = A[2] / l;
    } else {
        B[0] = A[0];
        B[1] = A[1];
        B[2] = A[2];
    }
}

static inline void
normalizel(long double *A) {
    long double l = A[0]*A[0] + A[1]*A[1] + A[2]*A[2];
    if (l != 1.0L) {
        l = sqrtl(l);
        A[0] /= l;
        A[1] /= l;
        A[2] /= l;
    }
}

static inline long double
dotl(const long double *A, const long double *B) {
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
}

static inline long double
signl(const long double A) {
    return (A == 0.0L) ? 0.0L : ((A < 0.0L) ? -1.0L : 1.0L);
}

static inline int
equalsl(const long double *A, const long double *B) {
    return A[0] == B[0] && A[1] == B[1] && A[2] == B[2];
}

static inline void
multl(long double *T, const long double f) {
    T[0] *= f;
    T[1] *= f;
    T[2] *= f;
}

static inline long double
lengthl(long double *A, long double *B) {
    long double s;

    /* Special case for "exactly equal" that avoids all of the calculation. */
    if (equalsl(A, B)) {
        return 0.0L;
    }

    s = dotl(A, B);

    /* clip s to range -1.0 to 1.0 */
    s = (s < -1.0L) ? -1.0L : (s > 1.0L) ? 1.0L : s;

    return acosl(s);
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
  normalize
*/

char *normalize_signature = "(i)->(i)";

static void
DOUBLE_normalize(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    long double IN[3];
    double OUT[3];

    INIT_OUTER_LOOP_2
    intp is1=steps[0], is2=steps[1];
    BEGIN_OUTER_LOOP_2
        char *ip1=args[0], *op=args[1];

        load_pointl(ip1, is1, IN);

        normalize_output(IN, OUT);

        save_point(OUT, op, is2);
    END_OUTER_LOOP
}

static PyUFuncGenericFunction normalize_functions[] = { DOUBLE_normalize };
static void * normalize_data[] = { (void *)NULL };
static char normalize_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  cross
*/

char *cross_signature = "(i),(i)->(i)";

static void
DOUBLE_cross(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    long double A[3];
    long double B[3];
    long double C[3];

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1], is3=steps[2];
    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_pointl(ip1, is1, A);
        load_pointl(ip2, is2, B);

        crossl(A, B, C);

        save_pointl(C, op, is3);
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
    long double A[3];
    long double B[3];
    long double C[3];

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1], is3=steps[2];
    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_pointl(ip1, is1, A);
        load_pointl(ip2, is2, B);

        crossl(A, B, C);
        normalizel(C);

        save_pointl(C, op, is3);
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
    long double A[3];
    long double B[3];
    long double C[3];
    long double D[3];

    long double ABX[3];
    long double CDX[3];
    long double T[3];
    long double tmp[3];

    double nans[3];

    long double s;
    int match;

    nans[0] = nans[1] = nans[2] = NPY_NAN;

    INIT_OUTER_LOOP_5
    intp is1=steps[0], is2=steps[1], is3=steps[2], is4=steps[3], is5=steps[4];
    BEGIN_OUTER_LOOP_5
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *ip4=args[3], *op=args[4];

        load_pointl(ip1, is1, A);
        load_pointl(ip2, is2, B);
        load_pointl(ip3, is3, C);
        load_pointl(ip4, is4, D);

        match = !(equalsl(A, C) | equalsl(A, D) | equalsl(B, C) | equalsl(B, D));

        if (match) {
            crossl(A, B, ABX);
            crossl(C, D, CDX);
            crossl(ABX, CDX, T);
            normalizel(T);

            match = 0;
            crossl(ABX, A, tmp);
            s = signl(dotl(tmp, T));
            crossl(B, ABX, tmp);
            if (s == signl(dotl(tmp, T))) {
                crossl(CDX, C, tmp);
                if (s == signl(dotl(tmp, T))) {
                    crossl(D, CDX, tmp);
                    if (s == signl(dotl(tmp, T))) {
                        match = 1;
                    }
                }
            }
        }

        if (match) {
            multl(T, s);
            save_pointl(T, op, is5);
        } else {
            save_point(nans, op, is5);
        }
    END_OUTER_LOOP
}

static PyUFuncGenericFunction intersection_functions[] = { DOUBLE_intersection };
static void * intersection_data[] = { (void *)NULL };
static char intersection_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  length
*/

char *length_signature = "(i),(i)->()";

/*
 * Finds the length of the given great circle arc AB
 */
static void
DOUBLE_length(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    long double A[3];
    long double B[3];

    long double s;

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1];
    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_pointl(ip1, is1, A);
        load_pointl(ip2, is2, B);

        normalizel(A);
        normalizel(B);

        s = lengthl(A, B);

        *((double *)op) = (double)s;
    END_OUTER_LOOP
}

static PyUFuncGenericFunction length_functions[] = { DOUBLE_length };
static void * length_data[] = { (void *)NULL };
static char length_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  length
*/

char *intersects_point_signature = "(i),(i),(i)->()";

/*
 * Returns True is if the point C intersects arc AB
 */
static void
DOUBLE_intersects_point(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{

    long double A[3];
    long double B[3];
    long double C[3];

    long double total;
    long double left;
    long double right;
    long double diff;

    INIT_OUTER_LOOP_4
    intp is1=steps[0], is2=steps[1], is3=steps[2];
    BEGIN_OUTER_LOOP_4
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *op=args[3];

        load_pointl(ip1, is1, A);
        load_pointl(ip2, is2, B);
        load_pointl(ip3, is3, C);

        normalizel(A);
        normalizel(B);
        normalizel(C);

        total = lengthl(A, B);
        left = lengthl(A, C);
        right = lengthl(C, B);

        diff = fabsl((left + right) - total);

        *((uint8_t *)op) = diff < 1e-10L;
    END_OUTER_LOOP
}

static PyUFuncGenericFunction intersects_point_functions[] = { DOUBLE_intersects_point };
static void * intersects_point_data[] = { (void *)NULL };
static char intersects_point_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_BOOL };

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
        normalize_functions, normalize_data, normalize_signatures, 2, 1, 1,
        PyUFunc_None, "normalize",
        "inner on the last dimension and broadcast on the rest \n"      \
        "     \"(i)->(i)\" \n",
        0, normalize_signature);
    PyDict_SetItemString(dictionary, "normalize", f);
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

    f = PyUFunc_FromFuncAndDataAndSignature(
        length_functions, length_data, length_signatures, 2, 2, 1,
        PyUFunc_None, "length",
        "length of great circle arc \n" \
        "     \"(i),(i)->()\" \n",
        0, length_signature);
    PyDict_SetItemString(dictionary, "length", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        intersects_point_functions, intersects_point_data, intersects_point_signatures, 2, 3, 1,
        PyUFunc_None, "intersects_point",
        "True where point C intersects arc AB \n" \
        "     \"(i),(i),(i)->()\" \n",
        0, intersects_point_signature);
    PyDict_SetItemString(dictionary, "intersects_point", f);
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
