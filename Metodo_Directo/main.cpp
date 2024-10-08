/** 
 * @file main.cpp
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * The main file of the cuda implementation, it must be a cpp file.
 * Does the same as main.c but with some of cpp sintax.
 * 
 * It checks arguments, creates a simulation using the provided starting configuration file
 * runs the simulation and frees it when its done.
 * 
 * @param T (double): The internal time that the simulation should last measured in seconds
 * @param filepath (string): The filepath to the starting configuration file of the simulation, must be a csv or bin file
 */

#include <iostream>
#include <cstdlib>
#include "simulation.h"

int main(int argc, char **argv )
{   
    // Check command line arguments
    if(argc != 3) 
    {
        std::cout << "Invalid number of arguments\nUSE: ./main [T] [filepath]\n";
        return STATUS_ERROR;
    }
    
    //Run simulation for T seconds
    if(run_simulation(std::strtod(argv[1], nullptr), argv[2]) == STATUS_ERROR)
    {
        return STATUS_ERROR;
    }
    
    return STATUS_OK;
}