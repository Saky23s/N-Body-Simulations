#ifndef METODO_DIRECTO_DEFS_H
#define METODO_DIRECTO_DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "simulation.h"

//Define real
typedef double real, *realptr;

/**
 * @struct Simulation
 * @brief Structure with the information for the generation of the simulation of N bodies
 *
 * Structure declaration for the simulation, structured in the form
 * that the data is optimized to minimize cache misses 
 */
typedef struct _Simulation
{   
    //Bodies variables
    realptr masses;
    realptr positions;
    realptr velocity;
    realptr acceleration;
    
    int n;
    real half_dt;
    real tnow;
    real tout;

} _Simulation;

#endif

