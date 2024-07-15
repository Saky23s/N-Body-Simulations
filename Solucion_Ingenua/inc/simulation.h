#ifndef SIMULATION_H
#define SIMULATION_H

typedef struct _Simulation Simulation;
#define G 1
//#define G 6.674299999999999e-08 
#define dt 0.01
#define speed 0.01
#define softening 0.1

Simulation* load_bodies(char* filepath);
void free_simulation(Simulation* simulation);
void print_simulation_values(Simulation* simulation);
double run_simulation(Simulation* simulation, double T);

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

#define STATUS_ERROR 0
#define STATUS_OK 1

#endif

