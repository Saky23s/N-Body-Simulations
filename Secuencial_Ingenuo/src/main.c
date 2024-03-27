#include <stdio.h>
#include <stdlib.h>

#include "../inc/simulation.h"

#define G 1
#define dt 0.01
#define speed 0.05
#define softening 0.1

int main(int argc, char **argv )
{   
    //Check command line arguments
    if(argc != 3)
    {
        printf("Invalid number of arguments\nUSE: ./main [T] [filepath]\n");
        return -1;
    }

    //Create simulation from starting file
    Simulation *simulation = load_bodies(argv[2]);    
    print_simulation_values(simulation);
    free_simulation(simulation);
}