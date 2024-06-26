/** 
 * @file simulation_OpenMP_v1.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * File that implements the calculate_acceleration function using OpenMP to calculate the values using all of the CPU cores. 
 * 
 * @extends simulation.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../inc/simulation.h"
#include "simulation.c"

int calculate_acceleration(Simulation* simulation, double*k)
/**
 * Funtion to calculate the velocity and acceleration of the bodies using the current positions and velocities
 * @param simulation (Simulation*): a pointer to the simulation object we are simulating, in the holder variable the information must be stored as an array of values order as x1,y1,z1,vx1,vz1,vz1,x2,y2,z2,vx2,vz2,vz2...xn,yn,zn,vxn,vzn,vzn
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
**/
{   
    //Error checking
    if(simulation == NULL || k == NULL)
        return STATUS_ERROR;

    //Init values of k as 0
    for(int i = 0; i < simulation->n * 6; i++)
    {
        k[i] = 0.0;
    }

    //For all of the bodies, in parallel
    #pragma omp parallel for
    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 6;
        k[ioffset] = dt * simulation->holder[ioffset+3]; //vx
        k[ioffset+1] = dt * simulation->holder[ioffset+4]; //vy
        k[ioffset+2] = dt * simulation->holder[ioffset+5]; //vz
        //For all other bodies
        for(int j = 0; j < simulation->n; j++)
        {   
            //i and j cant be the same body
            if(i==j)
                continue;

            int joffset = j * 6;
            double dx = simulation->holder[joffset] - simulation->holder[ioffset]; //rx body 2 - rx body 1
            double dy = simulation->holder[joffset+1] - simulation->holder[ioffset+1]; //ry body 2 - ry body 1
            double dz = simulation->holder[joffset+2] - simulation->holder[ioffset+2]; //rz body 2 - rz body 1
            
            double r = pow(pow(dx, 2.0) + pow(dy, 2.0) + pow(dz, 2.0) + pow(softening, 2.0), 1.5); //distance magnitud with some softening
            
            k[ioffset+3] += dt * (G * simulation->masses[j] / r) * dx; //Acceleration formula for x
            k[ioffset+4] += dt * (G * simulation->masses[j] / r) * dy; //Acceleration formula for y
            k[ioffset+5] += dt * (G * simulation->masses[j] / r) * dz; //Acceleration formula for z
        }
    }
    return STATUS_OK;
}

