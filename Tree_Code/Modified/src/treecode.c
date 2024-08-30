/** 
 * @file treecode.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * Main file for the hierarchical N-body code.
 * 
 * Modifies the original work of Joshua E. Barnes to remove features
 * that are not required for this investigation 
 * 
 * Removed the use of getparams from this file, changed main to another file, calculate
 * real time duration, added fancy print of progress
 * 
 * Added memory free of the tree
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/
#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#define global                                  //don't default to extern 
#include "../inc/treecode.h"
#include <sys/time.h>

//Internal helpers
local int treeforce(void);
local int leapfrog(void);

double run_simulation(double T, char* filename)
/**
 * Funtion that will create the simulation using the configuration given at filename 
 * and will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)
 *
 * This funtion will calculate the positions of the bodies every in timesteps of 'dt' using the leapfrog method
 * and store them in /dev/shm/data/ as bin files every 'speed' seconds. Speed being a macro in treedefs.h
 * 
 * @param T (double) pointer to the simulation object with the initial values       
 * @param filename (char*): The filepath to the starting configuration file of the simulation, must be a csv or bin file
 * 
 * @return t (double): Real time that the simulation was running or STATUS_ERROR (0) in case of error
**/
{   
    //Error checking
    if(T <= 0.0)
    {
        error("Simulation end time must be positive");
        return STATUS_ERROR;
    }

    if(filename == NULL)
        return STATUS_ERROR;
    
    //Set stopping time    
    tstop = T;  
    
    //Calculate the total number of steps needed
    int steps = tstop / dt; 

    //Read initial data      
    if(load_bodies(filename) == STATUS_ERROR)
        return STATUS_ERROR;                        

    //Internal variables to measure time 
    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    //Do a complete calculation
    if(treeforce() == STATUS_ERROR)
    {
        freetree(bodytab);
        return STATUS_ERROR;
    }

    //And save it
    if(output() == STATUS_ERROR)
    {
        freetree(bodytab);
        return STATUS_ERROR;
    }
    
    //While not past tstop
    while (tstop - tnow > 0.01 * dt) 
    {   
        //Advance step using leapfrog
        if(leapfrog() == STATUS_ERROR)
        {
            freetree(bodytab);
            return STATUS_ERROR;
        }

        //Output results
        if(output() == STATUS_ERROR)
        {
            freetree(bodytab);
            return STATUS_ERROR;
        }

        //Print fancy progress 
        printf("\rIntegrating: step = %d / %d", nstep, steps);
	    fflush(stdout);
    }

    //Calculate how long the simulation took
    gettimeofday ( &t_end, NULL );
    printf("\nSimulation completed in %lf seconds\n",  WALLTIME(t_end) - WALLTIME(t_start));

    //Free memory of the tree
    freetree(bodytab);

    //Return the time that the simulation took
    return WALLTIME(t_end) - WALLTIME(t_start);
}

local int treeforce(void)
/**
 * This fuction will calculate the acceleration forces of the N bodies using the current positions
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    bodyptr p;

    //Set update to true in all bodies
    for (p = bodytab; p < bodytab+nbody; p++)
    {
        Update(p) = TRUE;
    }
    
    //Construct tree structure
    maketree(bodytab, nbody);

    //Compute initial forces
    gravcalc();

    //If diagnostics are required print the force stadistics
    #ifdef DIAGNOSTICS
        forcereport();
    #endif

    return STATUS_OK;
}

local int leapfrog(void)
/**
 * This funtion will calculate the next values of the simulation using the leapfrog numerical method
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 **/
{
    bodyptr p;

    for (p = bodytab; p < bodytab+nbody; p++) 
    {   
        ///Half step velocity
        //  vn+1/2 = vn + 1/2dt * a(rn)
        ADDMULVS(Vel(p), Acc(p), 0.5 * dt);

        //Use that velocity to full step the position
        //  rn+1 = rn + dt*vn+1/2
        ADDMULVS(Pos(p), Vel(p), dt);
    }

    //Calculate the accelerations with half step velocity and full step position
    if(treeforce() == STATUS_ERROR)
        return STATUS_ERROR;
    

    for (p = bodytab; p < bodytab+nbody; p++) 
    { 
        //Half step the velocity again (making a full step)
        //  vn+1 = vn+1/2 + 1/2dt * a(rn+1)
        ADDMULVS(Vel(p), Acc(p), 0.5 * dt); 
    }

    //Count another time step
    nstep++;

    //Advance time
    tnow = tnow + dt;

    return STATUS_OK;
}



