/** 
 * @file mathfns.h
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * header file for system and zeno math functions; assumes role
 * of math.h.

 * The original worked with real values, meaning that they can handle
 * either floats or doubles, depending on compiler switches.
 * 
 * This was reduced to keep only the necessary for this investigation,
 * this is only working with double and only the funtions that 
 * are used in the code 
 * 
 * At the end only rsqr was left
 *  
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#ifndef _mathfns_h
#define _mathfns_h

#include <math.h>

/**
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

//Funtion defined in mathfns. Only rsqr needed
#define rsqr     sqr
real rsqr(real);

#endif  /* ! _mathfns_h */
