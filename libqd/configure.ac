#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
define([QD_PATCH_VERSION], 23)
AC_PREREQ([2.69])
AC_INIT([qd],[2.3.QD_PATCH_VERSION],[dhbailey@lbl.gov])
AC_CONFIG_SRCDIR([src/qd_real.cpp])
AC_CONFIG_MACRO_DIR([m4])

PACKAGE="qd"
MAJOR_VERSION="2"
MINOR_VERSION="3"
PATCH_VERSION="QD_PATCH_VERSION"
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$PATCH_VERSION
AC_DEFINE_UNQUOTED([MAJOR_VERSION],$MAJOR_VERSION,[qd major version number])
AC_DEFINE_UNQUOTED([MINOR_VERSION],$MINOR_VERSION,[qd minor version number])
AC_DEFINE_UNQUOTED([PATCH_VERSION],$PATCH_VERSION,[qd patch number (sub minor version)])
echo "configuring $PACKAGE $VERSION..."

AC_CONFIG_AUX_DIR([config])
AM_INIT_AUTOMAKE([foreign tar-ustar])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_HEADERS([include/qd/qd_config.h])

AC_CANONICAL_HOST

# Checks for arguments
# --enable-inline
AC_ARG_ENABLE([inline], AS_HELP_STRING(--enable-inline, [inline commonly used functions.  [[default=yes]]]))
AC_MSG_CHECKING([if commonly used function is to be inlined])
if test "$enable_inline" != "no"; then
  enable_inline="yes"
  AC_DEFINE([QD_INLINE], [1], [Define to 1 to inline commonly used functions.])
fi
AC_MSG_RESULT($enable_inline)

# --enable-ieee-add
AC_ARG_ENABLE([ieee-add], AS_HELP_STRING(--enable-ieee-add, [use addition that satisfies IEEE-style error bound instead of Cray-style error bound.  [[default=no]]]))
AC_MSG_CHECKING([if addition with IEEE-style error bound is to be used])
if test "$enable_ieee_add" = "yes"; then
  AC_DEFINE([QD_IEEE_ADD], [1], [Define to 1 to use additions with IEEE-style error bounds.])
else
  enable_ieee_add="no"
fi
AC_MSG_RESULT($enable_ieee_add)

# --enable-sloppy-mul
AC_ARG_ENABLE([sloppy-mul], AS_HELP_STRING(--enable-sloppy-mul, [use fast but slightly inaccurate multiplication.  [[default=yes]]]))
AC_MSG_CHECKING([if sloppy multiplication is to be used])
if test "$enable_sloppy_mul" != "no"; then
  enable_sloppy_mul="yes"
  AC_DEFINE([QD_SLOPPY_MUL], [1], [Define to 1 to use sloppy multiplication (which is faster by slightly inaccurate).])
fi
AC_MSG_RESULT($enable_sloppy_mul)

# --enable-sloppy-div
AC_ARG_ENABLE([sloppy-div], AS_HELP_STRING(--enable-sloppy-div, [use fast but slightly inaccurate division.  [[default=yes]]]))
AC_MSG_CHECKING([if sloppy division is to be used])
if test "$enable_sloppy_div" != "no"; then
  enable_sloppy_div="yes"
  AC_DEFINE([QD_SLOPPY_DIV], [1], [Define to 1 to use sloppy division (which is faster by slightly inaccurate).])
fi
AC_MSG_RESULT($enable_sloppy_div)


# Checks for programs.
AC_LANG(C++)
AC_PROG_CXX
AC_PROG_CC


# --enable-fma
AC_ARG_ENABLE([fma], AS_HELP_STRING(--enable-fma, [use fused multiply-add/subtract (auto,gnu,ia64,c99,ibm,compiler).  Use this option with care.  [[default=auto]]]))
if test "x$enable_fma" = "x"; then
  enable_fma="auto"
fi
case "$enable_fma" in
  yes)
    AX_CXX_FMA
    if test "x$ax_cxx_fma" = "x"; then
      AC_MSG_ERROR([Cannot find working fused multiply add.])
    fi
  ;;
  auto)
    case $host in
      powerpc*-*-*)
        AX_CXX_FMA(["ibm gnu"])
      ;;
      ia64-*)
        AX_CXX_FMA(["ia64 gnu"])
      ;;
    esac
  ;;
  no) 
    ax_cxx_fma=
  ;;
  *) 
    AX_CXX_FMA([$enable_fma])
    if test "x$ax_cxx_fma" = "x"; then
      AC_MSG_ERROR([Cannot find working fused multiply add.])
    fi
  ;;
esac
if test "x$ax_cxx_fma" = "x"; then
  ax_cxx_fma="none"
  ax_cxx_fms="none"
else
  AC_DEFINE_UNQUOTED([QD_FMA(x,y,z)], [$ax_cxx_fma], 
    [If fused multiply-add is available, define correct macro for using it.])
  AC_DEFINE_UNQUOTED([QD_FMS(x,y,z)], [$ax_cxx_fms], 
    [If fused multiply-subtract is available, define correct macro for using it.])
fi

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([ieeefp.h])

CONFIGURE_ARGS=$ac_configure_args
AC_SUBST(CONFIGURE_ARGS)

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_STRUCT_TM

# Check for x86 FPU fix
AC_MSG_CHECKING([whether x86 FPU control code is needed])
x86_fix="no"
case "$host" in
  i?86-*-* | k?*-*-* | athlon-*-* | x86_64-*-*)
    AC_DEFINE([X86], [1], [Whether to use x86 fpu fix.])
    x86_fix="yes"
  ;;
esac
AC_MSG_RESULT($x86_fix)
if test "$x86_fix" = "yes"; then
  AC_CHECK_HEADERS([fpu_control.h])
fi

# Checks for library functions.
AC_CHECK_FUNCS([gettimeofday])
AX_CXX_CLOCK_GETTIME
if test "$ax_cxx_clock_gettime" = "yes"; then
  AC_SEARCH_LIBS([clock_gettime], [rt], 
    [AC_DEFINE([HAVE_CLOCK_GETTIME], [1], 
      [Define to 1 if your system has the clock_gettime function.])])
fi
AX_CXX_ISNAN
AC_DEFINE_UNQUOTED([QD_ISNAN(x)], [$ax_cxx_isnan], [Define this macro to be the isnan(x) function.])
AX_CXX_ISINF
AC_DEFINE_UNQUOTED([QD_ISINF(x)], [$ax_cxx_isinf], [Define this macro to be the isinf(x) function.])
AX_CXX_ISFINITE
AC_DEFINE_UNQUOTED([QD_ISFINITE(x)], [$ax_cxx_isfinite], [Define this macro to be the isfinite(x) function.])
AX_CXX_COPYSIGN
AC_DEFINE_UNQUOTED([QD_COPYSIGN(x, y)], [$ax_cxx_copysign], [Define this macro to be the copysign(x, y) function.])

AC_DEFINE([QD_API], [], [Any special symbols needed for exporting APIs.])

# check for C++ std includes
AC_DEFINE([QD_HAVE_STD], [1], [Define to 1 if your compiler have the C++ standard include files.])

# Check for fortran-90 compiler

AC_ARG_ENABLE([fortran], AS_HELP_STRING(--enable-fortran, [build Fortran 77/90 interfaces [[default=auto]]]))
if test "$enable_fortran" != "no"; then
  AC_LANG_PUSH(Fortran)
  AC_PROG_FC([xlf95 ifort pathf95 f95 gfortran g95 pgf95 lf95 fort ifc efc pathf90 xlf90 pgf90 epcf90 xlf f90])
  if test "$FC" != ""; then
    enable_fortran="yes"
    AC_DEFINE([HAVE_FORTRAN], [1], [Define to 1 if Fortran interface is to be compiled.])
    AC_SUBST(FC, "$FC")

    AC_FC_LIBRARY_LDFLAGS
    AC_FC_FREEFORM

    FC_PROG_NAME=`basename $FC`

    case $host in
      i?86-*-* | k?*-*-* | athlon-*-* | ia64-*-* | x86_64-*-*)
        case "$FC_PROG_NAME" in
          ifc|ifort|efc)
            REQ_FCFLAGS="$REQ_FCFLAGS -mp -Vaxlib"
          ;;
        esac
      ;;
      powerpc*-*-*)
        case "$FC_PROG_NAME" in
          xlf*) 
            REQ_FCFLAGS="$REQ_FCFLAGS -qstrict" 
          ;;
        esac
      ;;
    esac

    AX_FC_ETIME
    AC_SUBST(ETIME, $ax_fc_etime)
    AX_CXX_FC_LIB

    AC_FC_WRAPPERS
    AC_FC_MAIN
    AX_F90_MODULE_FLAG
    AC_SUBST(ax_f90_modflag)
    AX_F90_MODULE_STYLE
    module_case=`echo "$ax_f90_module_style" | sed 's/,.*$//'`
    module_ext=`echo "$ax_f90_module_style" | sed 's/^.*, //'`
    AC_SUBST(module_ext)
  else
    enable_fortran="no"
  fi
  AC_SUBST(REQ_FCFLAGS)
  FCFLAGS="$FCFLAGS $REQ_FCFLAGS"
  AC_LANG_POP(Fortran)
fi

AM_CONDITIONAL(UPCASE_MODULE, [test "$module_case" = "uppercase"])
AM_CONDITIONAL(HAVE_FORTRAN, [test "$enable_fortran" = "yes"])

AC_SUBST(F77, $FC)
AC_SUBST(FFLAGS, $FCFLAGS)

# Add libraries
AC_CHECK_LIB(m,sqrt)

# libtool stuff
LT_INIT

# Output
AC_CONFIG_FILES([Makefile src/Makefile include/Makefile
                 tests/Makefile fortran/Makefile qd-config
                 fortran/second.f qd.pc])
AC_OUTPUT
chmod +x qd-config

echo
echo "Using "
echo "  C++ Compiler      = $CXX"
echo "  C++ Flags         = $CXXFLAGS $DEFS"
if test "$enable_fortran" = "yes"; then
  echo "  F90 Compiler      = $FC"
  echo "  F90 Flags         = $FCFLAGS $FCFLAGS_f90"
  echo "  F90 Libs          = $FCLIBS"
fi
echo "  enable fortran    = $enable_fortran"
echo "  enable inline     = $enable_inline"
echo "  enable x86 fix    = $x86_fix"
echo "  enable ieee add   = $enable_ieee_add"
echo "  enable sloppy mul = $enable_sloppy_mul"
echo "  enable sloppy div = $enable_sloppy_div"
echo "  fma               = $ax_cxx_fma"
echo "  fms               = $ax_cxx_fms"

