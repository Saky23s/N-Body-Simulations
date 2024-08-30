/****************************************************************************/
/* MATHFNS.C: utility routines for various sorts of math operations. Most   */
/* these functions work with real values, meaning that they can handle      */
/* either floats or doubles, depending on compiler switches.                */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"

real rsqr(real x)
/**
 * Computes the square operation of a number as x*x
 * @param x (real): number to be squared
 * @return x*x (real): the number squared
 */
{
    return (x * x);
}












