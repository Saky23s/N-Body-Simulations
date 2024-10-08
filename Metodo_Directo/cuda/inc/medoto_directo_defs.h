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
    realptr x;
    realptr y;
    realptr z;
    realptr vx;
    realptr vy;
    realptr vz;
    int n;

    //Control variables
    real half_dt;
    real tnow;
    real tout;

    //Cuda variables
    realptr d_masses;
    realptr d_x;
    realptr d_y;
    realptr d_z;
    realptr d_ax;
    realptr d_ay;
    realptr d_az;
    realptr d_vx;
    realptr d_vy;
    realptr d_vz;
    realptr d_block_holder;
    dim3 threadBlockDims;
    dim3 gridDims;

} _Simulation;


#endif

