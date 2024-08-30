/****************************************************************************/
/* TREECODE.C: new hierarchical N-body code.                                */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
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

    tstop = strtod(argv[1], NULL);  
    
    //Read initial data      
    if(load_bodies(argv[2]) == STATUS_ERROR)
        return STATUS_ERROR;                        

    treeforce();                            /* do complete calculation  */
    output();                               /* and report diagnostics   */
    

    if (dt != 0.0)                           /* if time steps requested  */
        while (tstop - tnow > 0.01 * dt) 
        {   /* while not past tstop     */
            stepsystem();                       /* advance step by step     */
            output();                           /* and output results       */
        }

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

    #ifdef DIAGNOSTICS
        forcereport();                              /* print force statistics   */
    #endif
}

/*
 * STEPSYSTEM: advance N-body system using simple leap-frog.
 */

local void stepsystem(void)
{
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



