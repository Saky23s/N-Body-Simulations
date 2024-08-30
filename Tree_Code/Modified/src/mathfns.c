/** 
 * @file mathfns.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * The original file had various sorts of math operations. that 
 * worked with real values, meaning that they can handle
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












