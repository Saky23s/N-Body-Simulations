/** 
 * @file simulation_secuencial.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * File that implements the calculate_acceleration function in a secuential matter. 
 * 
 * @extends simulation.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "../inc/simulation.h"
#include "aux.c"

#define FILENAME_MAX_SIZE 256

/**
 * @struct Simulation
 * @brief Structure with the information for the generation of the simulation of N bodies
 *
 * Structure declaration for the simulation, structured in the form
 * that the data is optimized to minimize cache misses 
 */
struct _Simulation
{   
    //Bodies variables
    double* masses;
    double* positions;
    double* velocity;
    double* acceleration;
    
    int n;
    double half_dt;
} _Simulation;

//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int leapfrog(Simulation* simulation);
int save_values_csv(Simulation* simulation, char* filename); 
int save_values_bin(Simulation* simulation, char* filename);
int calculate_acceleration(Simulation* simulation);
double checkEnergy(Simulation* simulation);

double run_simulation(Simulation* simulation, double T)
/**
 * Funtion that will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)
 *
 * This funtion will calculate the positions of the bodies every in timesteps of 'dt' using the leapfrog method
 * and store them in /dev/shm/data/ as bin files every 'speed' seconds. Speed being a macro in simulation.h
 * 
 * @param simulation (Simulation*) pointer to the simulation object with the initial values       
 * @param T (double): Internal ending time of the simulation
 * 
 * @return t (double): Real time that the simulation was running or STATUS_ERROR (0) in case of error
**/
{   
    //Calculate the number of steps we will have to take to get to T
    long int steps = T / dt;
    //Calculate the number of timesteps we must do before saving the data
    long int save_step = speed / dt;
    //Internal variables to keep track of csv files written
    long int file_number = 1;

    char filename[FILENAME_MAX_SIZE];

    //double starting_energy = checkEnergy(simulation);
    
    printf("Simulating secuentially %d bodies\n", simulation->n);
    
    //Internal variables to measure time 
    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    //Run simulation
    for(long int step = 1; step <= steps; step++)
    {
        //Integrate next step using leapfrog
        if(leapfrog(simulation) == STATUS_ERROR)
            return STATUS_ERROR;
        
        //Save data if we must
        if(step % save_step == 0)
        {   
            if(snprintf(filename, FILENAME_MAX, "/dev/shm/data/%ld.bin", file_number) < 0)
                return STATUS_ERROR;
                
            if(save_values_bin(simulation, filename) == STATUS_ERROR)
                return STATUS_ERROR;
            file_number++;
        }

        //Print fancy progress 
        printf("\rIntegrating: step = %ld / %ld", step, steps);
	    fflush(stdout);
    }
    
    //Calculate how long the simulation took
    gettimeofday ( &t_end, NULL );
    printf("\nSimulation completed in %lf seconds\n",  WALLTIME(t_end) - WALLTIME(t_start));
    
    //For testing purposes only
    //printf("Energy drift = %lf\n", fabs(checkEnergy(simulation) - starting_energy)); 

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

    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 3;
        //Half step velocity
        //  vn+1/2 = vn + 1/2dt * a(rn)
        simulation->velocity[ioffset] = simulation->velocity[ioffset] + simulation->acceleration[ioffset] * simulation->half_dt;
        simulation->velocity[ioffset+1] = simulation->velocity[ioffset+1] + simulation->acceleration[ioffset+1] * simulation->half_dt;
        simulation->velocity[ioffset+2] = simulation->velocity[ioffset+2] + simulation->acceleration[ioffset+2] * simulation->half_dt;

        //Use that velocity to full step the position
        //  rn+1 = rn + dt*vn+1/2
        simulation->positions[ioffset] = simulation->positions[ioffset] + simulation->velocity[ioffset] * dt;
        simulation->positions[ioffset+1] = simulation->positions[ioffset+1] + simulation->velocity[ioffset+1] * dt;
        simulation->positions[ioffset+2] = simulation->positions[ioffset+2] + simulation->velocity[ioffset+2] * dt;
    }

    //Calculate the acceleratuions with half step velocity and full step position
    if(calculate_acceleration(simulation) == STATUS_ERROR)
        return STATUS_ERROR;

    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 3;
        
        //Half step the velocity again (making a full step)
        //  vn+1 = vn+1/2 + 1/2dt * a(rn+1)
        simulation->velocity[ioffset] = simulation->velocity[ioffset] + simulation->acceleration[ioffset] * simulation->half_dt;
        simulation->velocity[ioffset+1] = simulation->velocity[ioffset+1] + simulation->acceleration[ioffset+1] * simulation->half_dt;
        simulation->velocity[ioffset+2] = simulation->velocity[ioffset+2] + simulation->acceleration[ioffset+2] * simulation->half_dt;
    }

    return STATUS_OK;
}

int calculate_acceleration(Simulation* simulation)
/**
 * Funtion to calculate the acceleration of the N bodies using the current positions and velocities
 * @param simulation(Simulation*): a pointer to the simulation object we are simulating
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 *         The resulting acceleration is stored inside acceleration atribute of the simulation
**/
{   
    //Error checking
    if(simulation == NULL)
        return STATUS_ERROR;

    //Init values of k as 0
    for(int i = 0; i < simulation->n * 3; i++)
    {
        simulation->acceleration[i] = 0.0;
    }

    //For all of the bodies
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
            double dx = simulation->positions[joffset] - simulation->positions[ioffset]; //rx body 2 - rx body 1
            double dy = simulation->positions[joffset+1] - simulation->positions[ioffset+1]; //ry body 2 - ry body 1
            double dz = simulation->positions[joffset+2] - simulation->positions[ioffset+2]; //rz body 2 - rz body 1
            
            double r = pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(softening, 2); //distance with some softening

            simulation->acceleration[ioffset] += (G * simulation->masses[j] * dx) / pow(r, 1.5); //Acceleration formula for x
            simulation->acceleration[ioffset+1] += (G * simulation->masses[j] * dy) / pow(r, 1.5); //Acceleration formula for y
            simulation->acceleration[ioffset+2] += (G * simulation->masses[j] * dz) / pow(r, 1.5); //Acceleration formula for z
        }
    }

    return STATUS_OK;
}

Simulation* load_bodies(char* filepath)
/**
 * This funtion creates a new Simulation and fills it using the starting values from a file
 * @param filepath (char*):  a path to the file with the starting data, must be csv or bin file
 * @return simulation (Simulation*): a pointer to the new Simulation filled with the data in filepath
 */
{   
    //Allocate memory for the Simulation object itself
    Simulation* simulation = (Simulation*) malloc(sizeof(Simulation));
    if(simulation == NULL)
    {
        return NULL;
    }

    //Error checking
    if(filepath == NULL)
    {
        return NULL;
    }
    
    //Calculate half of dt so we dont have to do it every iteration
    simulation->half_dt = dt / 2.0;
    
    int extention_type = get_extention_type(filepath);
    if(extention_type == EXT_CSV)
    {   
        //Open file
        FILE* f = NULL;
        f = fopen(filepath, "r");
        if(f == NULL)
        {
            return NULL;
        }

        //Get the number of bodies by the number of lines minus the header
        simulation->n = count_lines_csv(f) - 1;
        if(simulation->n <= 0)
        {
            return NULL;
        }

        //Memory allocation for the arrays
        if(simulation_allocate_memory(simulation) != STATUS_OK)
        {
            return NULL;
        }

        //go back to the begining of file
        rewind(f);
        //For the number of bodies + header
        for(int i = 0; i < simulation->n + 1; i++)
        {     
            int j = i - 1;
            int joffset = j*3;
            //read header
            if(i == 0)
            {   
                //skip header line
                fscanf(f, "%*[^\n]\n");
                continue;
            }

            if(fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*f\n", &simulation->positions[joffset], &simulation->positions[joffset+1], &simulation->positions[joffset+2], &simulation->masses[j], &simulation->velocity[joffset], &simulation->velocity[joffset+1], &(simulation->velocity[joffset+2])) == EOF)
            {
                printf("Error reading %s\n", filepath);
                return NULL;
            }
            
        }
        //close file
        fclose(f);
    }
    else if (extention_type == EXT_BIN)
    {
        //Read as binary
        FILE* f = fopen(filepath, "rb");
        if(f == NULL)
        {
            return NULL;
        }

        //Get file size
        fseek(f, 0, SEEK_END); 
        long size = ftell(f); 
        fseek(f, 0, SEEK_SET);

        //The number of bodies is the size of the file / size of each body
        simulation->n = size / (sizeof(double) * 8); 

        //Memory allocation for the arrays
        if(simulation_allocate_memory(simulation) != STATUS_OK)
        {
            return NULL;
        }
        
        //Buffer for one body
        double buffer[8];
        //Read the whole file
        for (int i = 0; i < simulation->n; i++)
        {   
            int ioffset = i * 3;
            if(fread(buffer,sizeof(buffer),1,f) == 0)
                return NULL;
                
            simulation->positions[ioffset] = buffer[0];     //x
            simulation->positions[ioffset+1] = buffer[1];   //y
            simulation->positions[ioffset+2] = buffer[2];   //z
            simulation->masses[i] = buffer[3];              //mass
            simulation->velocity[ioffset] = buffer[4];      //vx
            simulation->velocity[ioffset+1] = buffer[5];    //vy
            simulation->velocity[ioffset+2] = buffer[6];    //vz

            //Buffer[7] is radius, currently useless for data, only useful for graphics
        }
        fclose(f);
        
    }
    else
    {
        return NULL;
    }
    
    //Return simulation
    return simulation;
}

int simulation_allocate_memory(Simulation* simulation)
/**
 * Funtion that allocates all of the internal arrays of the simulation
 * 
 * @param simulation (Simulation*): pointer to a fresh simulation in which all of the internal pointer still have to be allocated
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
*/
{   
    if(simulation == NULL || simulation->n <= 0)
        return STATUS_ERROR;

    simulation->masses = (double*) malloc (simulation->n * sizeof(simulation->masses[0]));
    simulation->positions = (double*) malloc (simulation->n * 3 * sizeof(simulation->positions[0]));
    simulation->velocity = (double*) malloc (simulation->n * 3 * sizeof(simulation->velocity[0]));
    simulation->acceleration = (double*) malloc (simulation->n * 3 * sizeof(simulation->acceleration[0]));
    
    if(simulation->masses == NULL || simulation->positions == NULL || simulation->velocity == NULL || simulation->acceleration == NULL)
    {
        return STATUS_ERROR;
    }

    return STATUS_OK;
}

void free_simulation(Simulation* simulation)
/**
 * This funtion frees all the memory used by the simulation
 * @param simulation (Simulation*):  a pointer to the simulation being set free
 */
{   
    //Frees all internal arrays
    free(simulation->masses);

    free(simulation->positions);
    free(simulation->velocity);
    free(simulation->acceleration);

    //Frees the simulation object itself
    free(simulation);
}

void print_simulation_values(Simulation* simulation)
/**
 * This function prints all of the values used in the simulation, used only for debugging purpuses
 * @param simulation (Simulation*):  a pointer to the simulation being printed
 */
{
    if(simulation == NULL)
    {
        return;
    }

    printf("Simulation Values...\nN: %d\n", simulation->n);
    printf("Masses: [");
    for(int i = 0; i < simulation->n; i++)
    {
        //Print last value
        if(i == simulation->n - 1)
            printf("%lf",simulation->masses[i]);
        //Print normal value
        else
            printf("%lf,",simulation->masses[i]);

    }
    printf("]\n");

    printf("Bodies: [");
    for(int i = 0; i < simulation->n; i++)
    {     
        int ioffset = i*3;
        //Print the last value from all bodies
        if(i == simulation->n - 1) 
            printf("%lf, %lf, %lf, %lf, %lf, %lf",simulation->positions[ioffset], simulation->positions[ioffset+1], simulation->positions[ioffset+2], simulation->velocity[ioffset], simulation->velocity[ioffset+1], simulation->velocity[ioffset+2]); 
        else
            printf("%lf, %lf, %lf, %lf, %lf, %lf, ",simulation->positions[ioffset], simulation->positions[ioffset+1], simulation->positions[ioffset+2], simulation->velocity[ioffset], simulation->velocity[ioffset+1], simulation->velocity[ioffset+2]); 
    }
    printf("]\n");
}

int save_values_csv(Simulation* simulation, char* filename)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a csv
 * @param simulation (Simulation*):  a pointer to the simulation being stored
 * @param file (char*) the filepath in which the data is going to be stored as csv
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{   
    //Error checking
    if(simulation == NULL || filename == NULL)
        return STATUS_ERROR;

    //Open file
    FILE* f = fopen(filename, "w");
    if(f == NULL)
        return STATUS_ERROR;

    //For all n bodies
    for(int i = 0; i < simulation->n; i++)
    {      
        //Print body as csv x,y,z
        int ioffset = i*3;
        if(fprintf(f, "%lf,%lf,%lf\n", simulation->positions[ioffset], simulation->positions[ioffset+1], simulation->positions[ioffset+2]) < 0)
            return STATUS_ERROR;
    }

    fclose(f);
    return STATUS_OK;
}

int save_values_bin(Simulation* simulation, char* filename)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a bin
 * @param simulation (Simulation*):  a pointer to the simulation being stored
 * @param file (char*) the filepath in which the data is going to be stored as bin
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{   
    //Error checking
    if(simulation == NULL || filename == NULL)
        return STATUS_ERROR;

    //Open file
    FILE* f = fopen(filename, "wb");
    if(f == NULL)
        return STATUS_ERROR;

    double buffer[3];

    //For all n bodies
    for(int i = 0; i < simulation->n; i++)
    {      
        int ioffset = i*3;

        buffer[0] = simulation->positions[ioffset];
        buffer[1] = simulation->positions[ioffset+1];
        buffer[2] =  simulation->positions[ioffset+2];
        
        //write body as bin x,y,z
        if(fwrite(buffer, sizeof(buffer), 1, f) == 0)
            return STATUS_ERROR;
    }

    fclose(f);
    return STATUS_OK;
}

double checkEnergy(Simulation* simulation)
/**
 * A funtion that calculates the total energy of the system at a given time
 * @param simulation (Simulation*): The simulation in the moment we want to calculate the total energy 
 * @return total_energy (double): the total energy of the system
 */
{   
    double total_energy = 0.0;
    
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
            double dx = simulation->positions[ioffset] - simulation->positions[joffset]; //rx body 2 - rx body 1
            double dy = simulation->positions[ioffset+1] - simulation->positions[joffset+1]; //ry body 2 - ry body 1
            double dz = simulation->positions[ioffset+2] - simulation->positions[joffset+2]; //rz body 2 - rz body 1

            
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            total_energy -= (G * simulation->masses[i] * simulation->masses[j]) / r;
        }
    }

    return total_energy;
}