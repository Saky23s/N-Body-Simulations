#include <stdio.h>
#include <stdlib.h>

#include "../inc/simulation.h"



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
    if(simulation == NULL)
    {   
        printf("Error while loading simulation\n");
        return -1;
    }
    
    //Run simulation for T seconds
    run_simulation(simulation, strtod(argv[1], NULL));

    //Free memory
    free_simulation(simulation);
}