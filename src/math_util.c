#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "numpy/npy_3kcompat.h"

#include "numpy/npy_math.h"

#include "qd/c_qd.h"

/*
  The intersects, length and intersects_point calculations use "double
  double" representation internally, as supported by libqd.  Emperical
  testing shows that this level of precision is required for finding all
  intersections in typical HST images that are rotated only slightly from one
  another.
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

typedef struct {
    double x[4];
} qd;

static NPY_INLINE void
load_point(const char *in, const intp s, double* out) {
    out[0] = (*(double *)in);
    in += s;
    out[1] = (*(double *)in);
    in += s;
    out[2] = (*(double *)in);
}

static NPY_INLINE void
load_point_qd(const char *in, const intp s, qd* out) {
    out[0].x[0] = (*(double *)in);
    out[0].x[1] = 0.0;
    out[0].x[2] = 0.0;
    out[0].x[3] = 0.0;
    in += s;
    out[1].x[0] = (*(double *)in);
    out[1].x[1] = 0.0;
    out[1].x[2] = 0.0;
    out[1].x[3] = 0.0;
    in += s;
    out[2].x[0] = (*(double *)in);
    out[2].x[1] = 0.0;
    out[2].x[2] = 0.0;
    out[2].x[3] = 0.0;
}

static NPY_INLINE void
save_point(const double* in, char *out, const intp s) {
    *(double *)out = in[0];
    out += s;
    *(double *)out = in[1];
    out += s;
    *(double *)out = in[2];
}

static NPY_INLINE void
save_point_qd(const qd* in, char *out, const intp s) {
    *(double *)out = in[0].x[0];
    out += s;
    *(double *)out = in[1].x[0];
    out += s;
    *(double *)out = in[2].x[0];
}

static NPY_INLINE void
cross_qd(const qd *A, const qd *B, qd *C) {
    double tmp1[4];
    double tmp2[4];

    c_qd_mul(A[1].x, B[2].x, tmp1);
    c_qd_mul(A[2].x, B[1].x, tmp2);
    c_qd_sub(tmp1, tmp2, C[0].x);

    c_qd_mul(A[2].x, B[0].x, tmp1);
    c_qd_mul(A[0].x, B[2].x, tmp2);
    c_qd_sub(tmp1, tmp2, C[1].x);

    c_qd_mul(A[0].x, B[1].x, tmp1);
    c_qd_mul(A[1].x, B[0].x, tmp2);
    c_qd_sub(tmp1, tmp2, C[2].x);
}

static NPY_INLINE int
normalize_qd(const qd *A, qd *B) {
    size_t i;

    double T[4][4];
    double l[4];

    for (i = 0; i < 3; ++i) {
        c_qd_sqr(A[i].x, T[i]);
    }

    c_qd_add(T[0], T[1], T[3]);
    c_qd_add(T[3], T[2], T[3]);

    if (T[3][0] < -0.0) {
        PyErr_SetString(PyExc_ValueError, "Domain error in sqrt");
        return 1;
    }

    c_qd_sqrt(T[3], l);
    for (i = 0; i < 3; ++i) {
        c_qd_div(A[i].x, l, B[i].x);
    }

    return 0;
}

static NPY_INLINE void
dot_qd(const qd *A, const qd *B, qd *C) {
    size_t i;
    double tmp[4][4];

    for (i = 0; i < 3; ++i) {
        c_qd_mul(A[i].x, B[i].x, tmp[i]);
    }

    c_qd_add(tmp[0], tmp[1], tmp[3]);
    c_qd_add(tmp[3], tmp[2], C->x);
}

static NPY_INLINE double
sign(const double A) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    return (A < 0) ? -1.0 : 1.0;
#else
    return npy_signbit(A) ? -1.0 : 1.0;
#endif
}

static NPY_INLINE int
equals_qd(const qd *A, const qd *B) {
    return memcmp(A, B, sizeof(qd) * 3) == 0;
}

static NPY_INLINE int
length_qd(const qd *A, const qd *B, qd *l) {
    qd s;

    /* Special case for "exactly equal" that avoids all of the calculation. */
    if (equals_qd(A, B)) {
        l->x[0] = 0.0;
        l->x[1] = 0.0;
        l->x[2] = 0.0;
        l->x[3] = 0.0;
        return 0;
    }

    dot_qd(A, B, &s);

    if (s.x[0] != s.x[0] ||
        s.x[0] < -1.0 ||
        s.x[0] > 1.0) {
        PyErr_SetString(PyExc_ValueError, "Out of domain for acos");
        return 1;
    }

    c_qd_acos(s.x, l->x);
    return 0;
}

static NPY_INLINE void
intersection_qd(const qd *A, const qd *B, const qd *C, const qd *D,
                qd *T, double *s, int *match) {
    qd ABX[3];
    qd CDX[3];
    qd tmp[3];
    qd dot;

    *match = !(equals_qd(A, C) | equals_qd(A, D) | equals_qd(B, C) | equals_qd(B, D));

    if (*match) {
        cross_qd(A, B, ABX);
        cross_qd(C, D, CDX);
        cross_qd(ABX, CDX, T);
        if (normalize_qd(T, T)) return;

        *match = 0;
        cross_qd(ABX, A, tmp);
        dot_qd(tmp, T, &dot);
        *s = sign(dot.x[0]);
        cross_qd(B, ABX, tmp);
        dot_qd(tmp, T, &dot);
        if (*s == sign(dot.x[0])) {
            cross_qd(CDX, C, tmp);
            dot_qd(tmp, T, &dot);
            if (*s == sign(dot.x[0])) {
                cross_qd(D, CDX, tmp);
                dot_qd(tmp, T, &dot);
                if (*s == sign(dot.x[0])) {
                    *match = 1;
                }
            }
        }
    }
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
    qd IN[3];
    qd OUT[3];
    unsigned int old_cw;

    INIT_OUTER_LOOP_2
    intp is1=steps[0], is2=steps[1];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_2
        char *ip1=args[0], *op=args[1];

        load_point_qd(ip1, is1, IN);

        if (normalize_qd(IN, OUT)) return;

        save_point_qd(OUT, op, is2);
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
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
    qd A[3];
    qd B[3];
    qd C[3];
    unsigned int old_cw;

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1], is3=steps[2];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);

        cross_qd(A, B, C);

        save_point_qd(C, op, is3);
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
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
    qd A[3];
    qd B[3];
    qd C[3];
    unsigned int old_cw;

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1], is3=steps[2];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);

        cross_qd(A, B, C);
        if (normalize_qd(C, C)) return;

        save_point_qd(C, op, is3);
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction cross_and_norm_functions[] = { DOUBLE_cross_and_norm };
static void * cross_and_norm_data[] = { (void *)NULL };
static char cross_and_norm_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  triple_product
*/

char *triple_product_signature = "(i),(i),(i)->()";

/*
 * Finds the triple_product at *B* between *A*, *B*,  and *C*.
 */
static void
DOUBLE_triple_product(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    qd A[3];
    qd B[3];
    qd C[3];

    qd ABX[3];
    qd prod;

    unsigned int old_cw;

    INIT_OUTER_LOOP_4
    intp is1=steps[0], is2=steps[1], is3=steps[2];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_4
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *op=args[3];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);
        load_point_qd(ip3, is3, C);

        cross_qd(A, B, ABX);
        dot_qd(ABX, C, &prod);

        *(double *)op = prod.x[0];
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction triple_product_functions[] = { DOUBLE_triple_product };
static void * triple_product_data[] = { (void *)NULL };
static char triple_product_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

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
    qd A[3];
    qd B[3];
    qd C[3];
    qd D[3];

    qd T[3];

    double nans[3];

    double s;
    int match;

    unsigned int old_cw;

    INIT_OUTER_LOOP_5
    intp is1=steps[0], is2=steps[1], is3=steps[2], is4=steps[3], is5=steps[4];

    nans[0] = nans[1] = nans[2] = NPY_NAN;

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_5
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *ip4=args[3], *op=args[4];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);
        load_point_qd(ip3, is3, C);
        load_point_qd(ip4, is4, D);

        intersection_qd(A, B, C, D, T, &s, &match);

        if (match) {
            T[0].x[0] *= s;
            T[1].x[0] *= s;
            T[2].x[0] *= s;
            save_point_qd(T, op, is5);
        } else {
            save_point(nans, op, is5);
        }
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction intersection_functions[] = { DOUBLE_intersection };
static void * intersection_data[] = { (void *)NULL };
static char intersection_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  intersects
*/

char *intersects_signature = "(i),(i),(i),(i)->()";

/*
 * Returns True where the great circle arcs AB and CD intersects
 */
static void
DOUBLE_intersects(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    qd A[3];
    qd B[3];
    qd C[3];
    qd D[3];

    qd T[3];

    double s;
    int match;

    unsigned int old_cw;

    INIT_OUTER_LOOP_5
    intp is1=steps[0], is2=steps[1], is3=steps[2], is4=steps[3];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_5
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *ip4=args[3], *op=args[4];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);
        load_point_qd(ip3, is3, C);
        load_point_qd(ip4, is4, D);

        intersection_qd(A, B, C, D, T, &s, &match);

        *((char *)op) = match;
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction intersects_functions[] = { DOUBLE_intersects };
static void * intersects_data[] = { (void *)NULL };
static char intersects_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_BOOL };

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
    qd A[3];
    qd B[3];

    qd s;

    unsigned int old_cw;

    INIT_OUTER_LOOP_3
    intp is1=steps[0], is2=steps[1];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_3
        char *ip1=args[0], *ip2=args[1], *op=args[2];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);

        if (normalize_qd(A, A)) return;
        if (normalize_qd(B, B)) return;

        if (length_qd(A, B, &s)) return;

        *((double *)op) = s.x[0];
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction length_functions[] = { DOUBLE_length };
static void * length_data[] = { (void *)NULL };
static char length_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

/*///////////////////////////////////////////////////////////////////////////
  intersects_point
*/

char *intersects_point_signature = "(i),(i),(i)->()";

/*
 * Returns True is if the point C intersects arc AB
 */
static void
DOUBLE_intersects_point(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    qd A[3];
    qd B[3];
    qd C[3];

    qd total;
    qd left;
    qd right;
    double t1[4], t2[4];
    int result;

    unsigned int old_cw;

    INIT_OUTER_LOOP_4
    intp is1=steps[0], is2=steps[1], is3=steps[2];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_4
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *op=args[3];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);
        load_point_qd(ip3, is3, C);

        if (normalize_qd(A, A)) return;
        if (normalize_qd(B, B)) return;
        if (normalize_qd(C, C)) return;

        if (length_qd(A, B, &total)) return;
        if (length_qd(A, C, &left)) return;
        if (length_qd(C, B, &right)) return;

        c_qd_add(left.x, right.x, t1);
        c_qd_sub(t1, total.x, t2);
        c_qd_abs(t2, t1);

        c_qd_comp_qd_d(t1, 1e-10, &result);
        *((npy_bool *)op) = (result == -1);
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction intersects_point_functions[] = { DOUBLE_intersects_point };
static void * intersects_point_data[] = { (void *)NULL };
static char intersects_point_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_BOOL };

/*///////////////////////////////////////////////////////////////////////////
  angle
*/

char *angle_signature = "(i),(i),(i)->()";

/*
 * Finds the angle at *B* between *AB* and *BC*.
 */
static void
DOUBLE_angle(char **args, intp *dimensions, intp *steps, void *NPY_UNUSED(func))
{
    qd A[3];
    qd B[3];
    qd C[3];

    qd ABX[3];
    qd BCX[3];
    qd TMP[3];
    qd X[3];

    qd diff;
    qd inner;
    qd angle;

    double dangle;

    unsigned int old_cw;

    INIT_OUTER_LOOP_4
    intp is1=steps[0], is2=steps[1], is3=steps[2];

    fpu_fix_start(&old_cw);

    BEGIN_OUTER_LOOP_4
        char *ip1=args[0], *ip2=args[1], *ip3=args[2], *op=args[3];

        load_point_qd(ip1, is1, A);
        load_point_qd(ip2, is2, B);
        load_point_qd(ip3, is3, C);

        cross_qd(A, B, TMP);
        cross_qd(B, TMP, ABX);

        if (normalize_qd(ABX, ABX)) return;

        cross_qd(C, B, TMP);
        cross_qd(B, TMP, BCX);

        if (normalize_qd(BCX, BCX)) return;

        cross_qd(ABX, BCX, X);

        if (normalize_qd(X, X)) return;

        dot_qd(B, X, &diff);
        dot_qd(ABX, BCX, &inner);

        if (inner.x[0] != inner.x[0] ||
            inner.x[0] < -1.0 ||
            inner.x[0] > 1.0) {
            PyErr_SetString(PyExc_ValueError, "Out of domain for acos");
            return;
        }

        c_qd_acos(inner.x, angle.x);
        dangle = angle.x[0];

        if (diff.x[0] < 0.0) {
            dangle = 2.0 * NPY_PI - dangle;
        }

        *((double *)op) = dangle;
    END_OUTER_LOOP

    fpu_fix_end(&old_cw);
}

static PyUFuncGenericFunction angle_functions[] = { DOUBLE_angle };
static void * angle_data[] = { (void *)NULL };
static char angle_signatures[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

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
        "Normalize the vector to the unit sphere. \n"      \
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
        triple_product_functions, triple_product_data, triple_product_signatures, 2, 3, 1,
        PyUFunc_None, "triple_product",
        "Calculate the triple_product between A, B and C.\n" \
        "     \"(i),(i),(i)->()\" \n",
        0, triple_product_signature);

    PyDict_SetItemString(dictionary, "triple_product", f);
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
        intersects_functions, intersects_data, intersects_signatures, 2, 4, 1,
        PyUFunc_None, "intersects",
        "true where AB intersects CD \n" \
        "     \"(i),(i),(i),(i)->()\" \n",
        0, intersects_signature);
    PyDict_SetItemString(dictionary, "intersects", f);
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

    f = PyUFunc_FromFuncAndDataAndSignature(
        angle_functions, angle_data, angle_signatures, 2, 3, 1,
        PyUFunc_None, "angle",
        "Calculate the angle at B between AB and BC.\n" \
        "     \"(i),(i),(i)->()\" \n",
        0, angle_signature);
    PyDict_SetItemString(dictionary, "angle", f);
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
