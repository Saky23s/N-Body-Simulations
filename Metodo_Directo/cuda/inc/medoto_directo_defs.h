#ifndef METODO_DIRECTO_DEFS_H
#define METODO_DIRECTO_DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "../../simulation.h"

/**
 * @struct Simulation
 * @brief Structure with the information for the generation of the N body simulation
 *
 * Structure declaration for the simulation, structured in the form
 * that the data is optimized to minimize cache misses in the CPU
 * and with all the data needed to use the GPU to do the calculations
 */
typedef struct _Simulation
{   
    //Bodies variables
    realptr masses;
    realptr positions;
    realptr velocity;

    int n;

    //Control variables
    real half_dt;
    real tnow;
    real tout;

    //Cuda variables
    realptr d_masses;
    realptr d_positions;
    realptr d_acceleration;
    realptr d_velocity;
    realptr d_block_holder;
    dim3 threadBlockDimsGrav;
    dim3 gridDimsGrav;

    dim3 threadBlockLeap;
    dim3 gridDimsLeap;

} _Simulation;


#endif

