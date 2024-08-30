/****************************************************************************/
/* TREECODE.H: define various things for treecode.c and treeio.c.           */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#ifndef _treecode_h
#define _treecode_h

#include "treedefs.h"

/*
 * Parameters, state variables, and diagnostics for N-body integration.
 */

global string infile;                   /* file name for snapshot input     */

global real tstop;                      /* time to stop calculation         */

global string headline;                 /* message describing calculation   */

global real tnow;                       /* current value of time            */

global real tout;                       /* time of next output              */

global int nstep;                       /* number of time-steps             */

global int nbody;                       /* number of bodies in system       */

global bodyptr bodytab;                 /* points to array of bodies        */

/*
 * Prototypes for I/O routines.
 */

int load_bodies(char* filename);               /* read initial data file           */
int output(void);                      /* perform output operation         */

#ifdef DIAGNOSTICS
void forcereport(void);                 /* report on force calculation      */
#endif

#endif /* ! _treecode_h */
