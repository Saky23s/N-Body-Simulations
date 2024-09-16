#ifndef SIMULATION_H
#define SIMULATION_H

typedef struct _Simulation Simulation;

#define G 1
//#define G 6.674299999999999e-08 
#define dt 0.1
#define speed 0.1
#define softening 0.1

double run_simulation(double T, char* filepath);

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

#define STATUS_ERROR 0
#define STATUS_OK 1

//Define real
typedef double real, *realptr;

#endif