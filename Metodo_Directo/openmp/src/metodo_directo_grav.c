/** 
 * @file metodo_directo_grav.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * File that does the operations for the acceletation of each body suffered by the effects of the gravitational pull of all other bodies
 * 
 * This is the point where most of the time of the simulation happens, optimizing this file even by a little has grat impact on the 
 * total performance
 * 
 * In this case we are doing the calculations using OpenMP and compiling yhis file with -O3
 */

#include "../inc/medoto_directo_defs.h"

int calculate_acceleration(Simulation* simulation)
/**
 * Funtion to calculate the acceleration of the N bodies using the current positions and velocities.
 * It calculates the accelerations in parallel using OpenMP
 * 
 * @param simulation(Simulation*): a pointer to the simulation object we are simulating
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 *         The resulting acceleration is stored inside acceleration atribute of the simulation
**/
{   
    //Error checking
    if(simulation == NULL)
        return STATUS_ERROR;

    //Init values of acceleration as 0
    for(int i = 0; i < simulation->n * 3; i++)
    {
        simulation->acceleration[i] = 0.0;
    }

    //Calculate the square of softening only once 
    real softening2 = softening * softening;

    //For all of the bodies, in parallel
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 3;
    
        //For all other bodies
        for(int j = 0; j < simulation->n; j++)
        {   
            //i and j cant be the same body
            if(i==j)
                continue;

            int joffset = j * 3;
            real dx = simulation->positions[joffset] - simulation->positions[ioffset]; //rx body 2 - rx body 1
            real dy = simulation->positions[joffset+1] - simulation->positions[ioffset+1]; //ry body 2 - ry body 1
            real dz = simulation->positions[joffset+2] - simulation->positions[ioffset+2]; //rz body 2 - rz body 1
            
            real r = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + softening2); //distance magnitud with some softening
            r = (G * simulation->masses[j] * r * r * r ); //Acceleration formula

            simulation->acceleration[ioffset] += r * dx; //Acceleration formula for x
            simulation->acceleration[ioffset+1] += r * dy; //Acceleration formula for y
            simulation->acceleration[ioffset+2] += r * dz; //Acceleration formula for z
        }
    }

    return STATUS_OK;
}