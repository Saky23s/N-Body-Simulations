/** 
 * @file time.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * This file will run and save the execution time of the simulation
 * 
 * It checks arguments, creates a simulation using the provided starting configuration file
 * runs the simulation and frees it when its done.
 * 
 * @param T (double): The internal time that the simulation should last measured in seconds
 * @param filepath (char *): The filepath to the starting configuration file of the simulation, must be a csv or bin file
 * @param savepath (char *): The filepath to append the ending time simulation
 */

#include <stdio.h>
#include <stdlib.h>
#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../inc/treedefs.h"

int main(int argc, char **argv )
{   
    //Check command line arguments
    if(argc != 4)
    {
        printf("Invalid number of arguments\nUSE: ./main [T] [filepath]\n");
        return STATUS_ERROR;
    }
    
    //Create and run simulation for T seconds
    double t = run_simulation(strtod(argv[1], NULL), argv[2]);
    FILE* output_file = fopen(argv[3], "a");
    if(output_file == NULL)
        return STATUS_ERROR;
        
    //Store time
    fprintf(output_file, " %lf\n", t);

    //Free memory
    fclose(output_file);
    return STATUS_OK;
}