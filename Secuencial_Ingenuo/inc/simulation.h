
typedef struct Simulation
{
    double* bodies;
    double* masses;
    int n;
} Simulation;

Simulation* load_bodies(char* filepath);
void free_simulation(Simulation* simulation);
void print_simulation_values(Simulation* simulation);