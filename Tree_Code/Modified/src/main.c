/** 
 * @file main.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * The main file of the secuential implementation of the tree code
 * 
 * It checks arguments, creates a simulation using the provided starting configuration file
 * runs the simulation and frees it when its done.
 * 
 * @param T (double): The internal time that the simulation should last measured in seconds
 * @param filepath (char *): The filepath to the starting configuration file of the simulation, must be a csv or bin file
 */

#include <stdio.h>
#include <stdlib.h>
#include "../inc/stdinc.h"

int main(int argc, char **argv )
{   
    //Check command line arguments
    if(argc != 3)
    {
        printf("Invalid number of arguments\nUSE: ./main [T] [filepath]\n");
        return STATUS_ERROR;
    }
    
    //Create and run simulation for T seconds
    if(run_simulation(strtod(argv[1], NULL), argv[2]) == STATUS_ERROR)
    {
        return STATUS_ERROR;
    }
    
    return STATUS_OK;
}