/** 
 * @file treecode.h
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * This document defines various things for treecode.c and treeio.c.
 * 
 * This document has been modified lightly to remove funtions not needed in this investigations
 * and to adapt it to work with our existing framework
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#ifndef _treecode_h
#define _treecode_h
#include "treedefs.h"

/***************************************/
//parametes, state variables and diagnostics for N-body integration.
//Time to stop calculation
global real tstop;

//Current value of time
global real tnow;

//Time of next output
global real tout;

//Number of time steps done 
global int nstep;

//Number of bodies in the system
global int nbody;

//Pointer to array of bodies
global bodyptr bodytab;

/***************************************/
// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

/***************************************/
//Prototypes for I/O routines.

//Read initial data file
int load_bodies(char* filename);

//Perform output operation
int output(void);

#ifdef DIAGNOSTICS
//Report on force calculation
void forcereport(void);
#endif

#endif /* ! _treecode_h */
