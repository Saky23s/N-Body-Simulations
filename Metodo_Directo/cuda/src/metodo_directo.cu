/** 
 * @file metodo_directo.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * Main file for the direct method, it is in charge of connecting all the funtions from the other files,
 * it loads the simulations using the methods from metodo_directo_io, runs the simulation 
 * using the leapfrog integration method and the gravity calculation from metodo_directo_grav and when finished running it free resources
 * 
 * If variable DIAGNOSTICS is defined in the makefile it will show the energy of the system
 */

#include "../inc/medoto_directo_defs.h"
#include "../inc/metodo_directo.h"

//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int leapfrog(Simulation* simulation);

//cuda kernels
__global__ void leapfrog_step(int n, realptr vx, realptr vy, realptr vz, realptr ax, realptr ay, realptr az, realptr x, realptr y, realptr z, real half_dt, real d_dt);
__global__ void leapfrog_halfstep(int n, realptr vx, realptr vy, realptr vz, realptr ax, realptr ay, realptr az, real half_dt);

#ifdef DIAGNOSTICS
real checkEnergy(Simulation* simulation);
#endif

double run_simulation(double T, char* filename)
/**
 * Funtion that will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)
 *
 * This funtion will calculate the positions of the bodies every in timesteps of 'dt' using the leapfrog method
 * and store them in /dev/shm/data/ as bin files every 'speed' seconds. Speed being a macro in simulation.h
 *       
 * @param T (double): Internal ending time of the simulation
 * @param filename (char*): The filepath to the starting configuration file of the simulation, must be a csv or bin file 
 * 
 * @return t (double): Real time that the simulation was running or STATUS_ERROR (0) in case of error
**/
{   
    //Read initial data  
    Simulation* simulation = NULL;
    simulation = load_bodies(filename);

    //Check that it loaded correctly
    if(simulation == NULL)
        return STATUS_ERROR;  

    //Print starting message
    printf("Simulating with CUDA %d bodies\n", simulation->n);
    
    //Calculate the number of steps we will have to take to get to T
    int steps = T / dt;

    //In case of doing diagnostics, calculate starting energy
    #ifdef DIAGNOSTICS
    real starting_energy = checkEnergy(simulation);
    #endif

    //Number of steps taken
    int nstep = 0;

    //Filenumber count
    int filenumber = 0;

    //Internal variables to measure time 
    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    //Do a complete calculation
    if(calculate_acceleration(simulation) == STATUS_ERROR)
    {
        free_simulation(simulation);
        return STATUS_ERROR;
    }

    //And save it
    if(output(simulation, &filenumber) == STATUS_ERROR)
    {
        free_simulation(simulation);
        return STATUS_ERROR;
    }

    //Run simulation
    while (T - simulation->tnow > 0.01 * dt) 
    {
        //Integrate next step using leapfrog
        if(leapfrog(simulation) == STATUS_ERROR)
        {
            free_simulation(simulation);
            return STATUS_ERROR;
        }
        
        //Count another time step
        nstep++;

        //Advance time
        simulation->tnow = simulation->tnow + dt;

        //Save data if we must
        if(output(simulation, &filenumber) == STATUS_ERROR)
        {
            free_simulation(simulation);
            return STATUS_ERROR;
        }

        #ifndef DIAGNOSTICS
        //Print fancy progress 
        printf("\rIntegrating: step = %d / %d", nstep, steps);
	    fflush(stdout);
        #endif

        #ifdef DIAGNOSTICS
        //Print progress with energy, no fancy print
        printf("Integrating: step = %d / %d (Energy = %lf)\n", nstep, steps, fabs(checkEnergy(simulation)));
	    fflush(stdout);
        #endif
    }
    
    //Calculate how long the simulation took
    gettimeofday ( &t_end, NULL );
    printf("\nSimulation completed in %lf seconds\n",  WALLTIME(t_end) - WALLTIME(t_start));
    
    //For testing purposes only
    #ifdef DIAGNOSTICS
    printf("Energy drift = %lf\n", fabs(checkEnergy(simulation) - starting_energy)); 
    #endif

    //Free memory of the tree
    free_simulation(simulation);

    return WALLTIME(t_end) - WALLTIME(t_start);
}


int leapfrog(Simulation* simulation)
/**
 * This funtion will calculate the next values of the simulation using the leapfrog numerical method
 * 
 * @param simulation (Simulation*): a pointer to the simulation
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 **/
{       
    //Error checking
    if(simulation == NULL)
        return STATUS_ERROR;

    //Calculate the acceleration of the system
    if(calculate_acceleration(simulation) == STATUS_ERROR)
        return STATUS_ERROR;

    //Half step velocity and full step positions
    leapfrog_step<<<1,1024>>>(simulation->n, simulation->d_vx,  simulation->d_vy,  simulation->d_vz, simulation->d_ax,  simulation->d_ay, simulation->d_az, simulation->d_x, simulation->d_y, simulation->d_z, simulation->half_dt, dt);

    //Calculate the accelerations with half step velocity and full step position
    if(calculate_acceleration(simulation) == STATUS_ERROR)
        return STATUS_ERROR;

    leapfrog_halfstep<<<1,1024>>>(simulation->n, simulation->d_vx,  simulation->d_vy,  simulation->d_vz, simulation->d_ax,  simulation->d_ay, simulation->d_az, simulation->half_dt);

    return STATUS_OK;
}

__global__ void leapfrog_halfstep(int n, realptr vx, realptr vy, realptr vz, realptr ax, realptr ay, realptr az, real half_dt)
{   
    int index = threadIdx.x;
    int stride = blockDim.x;

    //Half step velocity
    //  vn+1/2 = vn + 1/2dt * a(rn)
    for (int i = index; i < n; i += stride)
    {
        vx[i] = vx[i] + ax[i] * half_dt;
    }

    //Half step velocity
    //  vn+1/2 = vn + 1/2dt * a(rn)
    for (int i = index; i < n; i += stride)
    {
        vy[i] = vy[i] + ay[i] * half_dt;
    }

    //Half step velocity
    //  vn+1/2 = vn + 1/2dt * a(rn)
    for (int i = index; i < n; i += stride)
    {
        vz[i] = vz[i] + az[i] * half_dt;
    }
    
}   

__global__ void leapfrog_step(int n, realptr vx, realptr vy, realptr vz, realptr ax, realptr ay, realptr az, realptr x, realptr y, realptr z, real half_dt, real d_dt)
{   
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        //  Half step velocity
        //  vn+1/2 = vn + 1/2dt * a(rn)
        vx[i] = vx[i] + ax[i] * half_dt;
        //Use that velocity to full step the position
        //  rn+1 = rn + dt*vn+1/2
        x[i] = x[i] + vx[i] * d_dt;
    }
    
    for (int i = index; i < n; i += stride)
    {
        //  Half step velocity
        //  vn+1/2 = vn + 1/2dt * a(rn)
        vy[i] = vy[i] + ay[i] * half_dt;
        //Use that velocity to full step the position
        //  rn+1 = rn + dt*vn+1/2
        y[i] = y[i] + vy[i] * d_dt;
    }

    for (int i = index; i < n; i += stride)
    {
        //  Half step velocity
        //  vn+1/2 = vn + 1/2dt * a(rn)
        vz[i] = vz[i] + az[i] * half_dt;
        //Use that velocity to full step the position
        //  rn+1 = rn + dt*vn+1/2
        z[i] = z[i] + vz[i] * d_dt;
    }
}

#ifdef DIAGNOSTICS
real checkEnergy(Simulation* simulation)
/**
 * A funtion that calculates the total energy of the system at a given time
 * @param simulation (Simulation*): The simulation in the moment we want to calculate the total energy 
 * @return total_energy (double): the total energy of the system
 */
{   
    real total_energy = 0.0;
    
    //Calculate kinetic energy
    for (int i = 0; i < simulation->n; i++)
    {   
        int ioffset = i*3;
        total_energy += 0.5*simulation->masses[i]*(simulation->velocity[ioffset]*simulation->velocity[ioffset]+ simulation->velocity[ioffset+1]* simulation->velocity[ioffset+1]+ simulation->velocity[ioffset+2]*simulation->velocity[ioffset+2]);
    }

    //Calculate potential energy
    for (int i = 0; i < simulation->n-1; i++)
    {   
        int ioffset = i*3;
        for(int j = i + 1; j < simulation->n; j++)
        {   
            int joffset = j * 3;
            real dx = simulation->positions[ioffset] - simulation->positions[joffset]; //rx body 2 - rx body 1
            real dy = simulation->positions[ioffset+1] - simulation->positions[joffset+1]; //ry body 2 - ry body 1
            real dz = simulation->positions[ioffset+2] - simulation->positions[joffset+2]; //rz body 2 - rz body 1

            
            real r = sqrt(dx * dx + dy * dy + dz * dz);
            total_energy -= (G * simulation->masses[i] * simulation->masses[j]) / r;
        }
    }

    return total_energy;
}
#endif