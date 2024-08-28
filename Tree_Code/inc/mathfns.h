/****************************************************************************/
/* MATHFNS.H: header file for system and zeno math functions; assumes role  */
/* of math.h.  Defines real-valued synonyms for system functions (eg, rsqrt */
/* for square root) and zeno functions (eg, seval)
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#ifndef _mathfns_h
#define _mathfns_h

#include <math.h>

/*
 * System math functions.  Use double-precision versions 
 */
#define rsqrt    sqrt
#define rcbrt    cbrt
#define rsin     sin
#define rcos     cos
#define rtan     tan
#define rasin    asin
#define racos    acos
#define ratan    atan
#define ratan2   atan2
#define rlog     log
#define rexp     exp
#define rlog10   log10
#define rsinh    sinh
#define rcosh    cosh
#define rtanh    tanh
#define rpow     pow
#define rabs     fabs
#define rfloor   floor
#define rceil    ceil

/*
 * Functions in mathfns.c; invoked just like those above.
 */
#define rsqr(x) x * x






#endif  /* ! _mathfns_h */
