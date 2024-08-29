/****************************************************************************/
/* TREECODE.C: new hierarchical N-body code.                                */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../inc/getparam.h"
#define global                                  /* don't default to extern  */
#include "../inc/treecode.h"


/* Prototypes for local procedures. */
local void treeforce(void);                     /* do force calculation     */
local void stepsystem(void);                    /* advance by one time-step */
local void startrun(void);                      /* initialize system state  */
local void testdata(void);                      /* generate test data       */

/*
 * MAIN: toplevel routine for hierarchical N-body code.
 */

int main(int argc, string argv[])
{   
    infile = argv[2];
    tstop = strtod(argv[1], NULL);

    startrun();                                 /* get params & input data  */
    startoutput();                              /* activate output code     */
    if (nstep == 0) {                           /* if data just initialized */
        treeforce();                            /* do complete calculation  */
        output();                               /* and report diagnostics   */
    }
#if defined(USEFREQ)
    if (freq != 0.0)                            /* if time steps requested  */
        while (tstop - tnow > 0.01/freq) {      /* while not past tstop     */
            stepsystem();                       /* advance step by step     */
            output();                           /* and output results       */
        }
#else
    if (dt != 0.0)                           /* if time steps requested  */
        while (tstop - tnow > 0.01 * dt) {   /* while not past tstop     */
            stepsystem();                       /* advance step by step     */
            output();                           /* and output results       */
        }
#endif
    return (0);                                 /* end with proper status   */
}

/*
 * TREEFORCE: common parts of force calculation.
 */

local void treeforce(void)
{
    bodyptr p;

    for (p = bodytab; p < bodytab+nbody; p++)   /* loop over all bodies     */
        Update(p) = TRUE;                       /* update all forces        */
    maketree(bodytab, nbody);                   /* construct tree structure */
    gravcalc();                                 /* compute initial forces   */
    forcereport();                              /* print force statistics   */
}

/*
 * STEPSYSTEM: advance N-body system using simple leap-frog.
 */

local void stepsystem(void)
{
#if defined(USEFREQ)
    real dt = 1.0 / freq;                    /* set basic time-step      */
#endif
    bodyptr p;

    for (p = bodytab; p < bodytab+nbody; p++) { /* loop over all bodies     */
        ADDMULVS(Vel(p), Acc(p), 0.5 * dt);  /* advance v by 1/2 step    */
        ADDMULVS(Pos(p), Vel(p), dt);        /* advance r by 1 step      */
    }
    treeforce();                                /* perform force calc.      */
    for (p = bodytab; p < bodytab+nbody; p++) { /* loop over all bodies     */
        ADDMULVS(Vel(p), Acc(p), 0.5 * dt);  /* advance v by 1/2 step    */
    }
    nstep++;                                    /* count another time step  */
    tnow = tnow + dt;                        /* finally, advance time    */
}

/*
 * STARTRUN: startup hierarchical N-body code.
 */

local void startrun(void)
{
    usequad = FALSE;

    options = "";
        
    inputdata();                        /* then read inital data    */
    
    rsize = 1.0;                            /* start root w/ unit cube  */
    nstep = 0;                              /* begin counting steps     */
    tout = tnow;                            /* schedule first output    */
     
}

