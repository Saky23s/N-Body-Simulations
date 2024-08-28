/** 
 * @file mathfns.h
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * Defines real-valued synonyms for system functions (eg, rsqrt 
 * for square root) and zeno functions (eg, seval)
 * 
 * This document has been modified lightly to remove funtions not needed in this investigations
 * and to adapt it to work with our existing framework
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#ifndef _mathfns_h
#define _mathfns_h

#include <math.h>

//System math functions.  Use double-precision versions 
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

//Square as two multiplications
#define rsqr(x) x * x

#endif  /* ! _mathfns_h */
