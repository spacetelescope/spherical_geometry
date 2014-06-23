/* This is a set of "safe" configuration for qd-library, that should
   work for most compilers we support, even if it's suboptimal.  To
   get optimal settings, it's best to use the system-built
   qd-library. */

#define QD_API
#define QD_INLINE 1
#define QD_HAVE_STD 1
#define QD_IEEE_ADD 1
#define QD_ISINF(x) ( (x) != 0.0 && (x) == 2.0 * (x) )
#define QD_ISFINITE(x) ( ((x) == 0.0) || ((x) != (2.0 * (x))) )
#define QD_ISNAN(x) ((x) != (x))
