/****************************************************************************/
/* MATHFNS.C: utility routines for various sorts of math operations. Most   */
/* these functions work with real values, meaning that they can handle      */
/* either floats or doubles, depending on compiler switches.                */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"

/*
 * RSQR, RQBE: compute x*x and x*x*x.
 */

real rsqr(real x)
{
    return (x * x);
}












