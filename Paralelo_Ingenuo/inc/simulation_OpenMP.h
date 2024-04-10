#ifndef SIMULATION_OPENMP_H
#define SIMULATION_OPENMP_H
typedef struct Simulation
{
    double* bodies;
    double* masses;
    int n;

    //Variable needed for internals
    double* k1;
    double* k2;
    double* k3;
    double* k4;
    double* holder;
} Simulation;


#define G 1
#define dt 0.01
#define speed 0.05
#define softening 0.1

Simulation* load_bodies(char* filepath);
void free_simulation(Simulation* simulation);
void print_simulation_values(Simulation* simulation);
double run_simulation(Simulation* simulation, double T);

#endif