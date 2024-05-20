#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<unistd.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "../inc/simulation.h"
#include "aux.c"

struct _Simulation
{
    double* bodies;
    double* masses;
    int n;

    //Variable needed for internals
    double* k1;
    double* k2;
    double* k3;
    double* k4;
    double* holder;
} _Simulation;



//Internal helpers
int save_values_csv(Simulation* simulation, char* filename);
int rk4(Simulation* simulation);
int save_values_bin(Simulation* simulation, char* filename);
int calculate_acceleration(Simulation* simulation, double*k);

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
        simulation->masses = (double*) malloc (simulation->n * sizeof(simulation->masses[0]));
        simulation->bodies = (double*) malloc ((simulation->n * 6)*sizeof(simulation->bodies[0]));
        simulation->k1 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k1[0]));
        simulation->k2 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k2[0]));
        simulation->k3 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k3[0]));
        simulation->k4 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k4[0]));
        simulation->holder = (double*) malloc ((simulation->n * 6)*sizeof(simulation->holder[0]));
        
        if(simulation->bodies == NULL || simulation->masses == NULL || simulation->k1 == NULL || simulation->k2 == NULL || simulation->k3 == NULL || simulation->k4 == NULL || simulation->holder == NULL)
        {
            return NULL;
        }

        //go back to the begining of file
        rewind(f);
        //For the number of bodies + header
        for(int i = 0; i < simulation->n + 1; i++)
        {   
            int j = i - 1;
            int joffset = j*6;
            //read header
            if(i == 0)
            {   
                //skip header line
                fscanf(f, "%*[^\n]\n");
                continue;
            }
            
            fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*f\n", &simulation->bodies[joffset], &simulation->bodies[joffset+1], &simulation->bodies[joffset+2], &simulation->masses[j], &simulation->bodies[joffset+3], &simulation->bodies[joffset+4], &(simulation->bodies[joffset+5]));
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
        simulation->masses = (double*) malloc (simulation->n * sizeof(simulation->masses[0]));
        simulation->bodies = (double*) malloc ((simulation->n * 6)*sizeof(simulation->bodies[0]));
        simulation->k1 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k1[0]));
        simulation->k2 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k2[0]));
        simulation->k3 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k3[0]));
        simulation->k4 = (double*) malloc ((simulation->n * 6)*sizeof(simulation->k4[0]));
        simulation->holder = (double*) malloc ((simulation->n * 6)*sizeof(simulation->holder[0]));
        
        if(simulation->bodies == NULL || simulation->masses == NULL || simulation->k1 == NULL || simulation->k2 == NULL || simulation->k3 == NULL || simulation->k4 == NULL || simulation->holder == NULL)
        {
            return NULL;
        }

        //Buffer for one body
        double buffer[8];
        //Read the whole file
        for (int i = 0; i < simulation->n; i++)
        {   
            int ioffset = i * 6;
            fread(buffer,sizeof(buffer),1,f);

            simulation->bodies[ioffset] = buffer[0];      //x
            simulation->bodies[ioffset+1] = buffer[1];    //y  
            simulation->bodies[ioffset+2] = buffer[2];    //z
            simulation->bodies[ioffset+3] = buffer[4];    //vx
            simulation->bodies[ioffset+4] = buffer[5];    //vy
            simulation->bodies[ioffset+5] = buffer[6];    //vz

            simulation->masses[i] = buffer[3];         //mass

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

double run_simulation(Simulation* simulation, double T)
/**
 * Funtion that will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)
 *
 * This funtion will calculate the positions of the bodies every in timesteps of 'dt'using the runge-kutta method
 * and store them in data/ as csv files every 'speed' seconds
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

    char filename[256];

    //Internal variables to measure time 
    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );


    //Run simulation
    for(long int step = 1; step <= steps; step++)
    {
        //Integrate next step using runge-kutta
        if(rk4(simulation) == STATUS_ERROR)
            return STATUS_ERROR;
        
        //Save data if we must
        if(step % save_step == 0)
        {   
            sprintf(filename, "../Graphics/data/%ld.bin", file_number);
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
    return WALLTIME(t_end) - WALLTIME(t_start);
}

int rk4(Simulation* simulation)
/**
 * This funtion will calculate the next values of the simulation using the runge-kutta method
 * 
 * @param simulation (Simulation*): a pointer to the simulation
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 **/
{   
    //Correctly set up holder
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->holder[i] = simulation->bodies[i];
    }
    
    //Calculate k1
    if(calculate_acceleration(simulation, simulation->k1) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+0.5*k1 to be able to calculate k2
    for(int i = 0; i < simulation->n*6; i++)
    {   
        simulation->holder[i] = simulation->bodies[i] + simulation->k1[i] * 0.5;
    }

    //Calculate k2
    if(calculate_acceleration(simulation, simulation->k2) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+0.5*k2 to be able to calculate k3
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->holder[i] = simulation->bodies[i] + simulation->k2[i] * 0.5;
    }

    //Calculate k3
    if(calculate_acceleration(simulation, simulation->k3) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+*k3 to be able to calculate k3
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->holder[i] = simulation->bodies[i] + simulation->k3[i];
    }

    //Calculate k4
    if(calculate_acceleration(simulation, simulation->k4) == STATUS_ERROR)
        return STATUS_ERROR;

    //Update simulation value to simulation.bodies + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->bodies[i] = simulation->bodies[i] + ((simulation->k1[i] + 2.0*simulation->k2[i] + 2.0*simulation->k3[i] + simulation->k4[i]) / 6.0);
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
    free(simulation->bodies);
    free(simulation->masses);
    free(simulation->k1);
    free(simulation->k2);
    free(simulation->k3);
    free(simulation->k4);
    free(simulation->holder);
    //Frees the simulation object itself
    free(simulation);
}

void print_simulation_values(Simulation* simulation)
/**
 * This funtion prints all of the valkues used by the simulation
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
    for(int i = 0; i < simulation->n * 6; i++)
    {   
        //Print the last value from all bodies
        if(i == (simulation->n * 6) - 1)
            printf("%lf",simulation->bodies[i]);
        //Print normal value
        else
            printf("%lf,",simulation->bodies[i]);
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
        int ioffset = i*6;
        fprintf(f, "%lf,%lf,%lf\n", simulation->bodies[ioffset], simulation->bodies[ioffset+1], simulation->bodies[ioffset+2]);
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
        int ioffset = i*6;

        buffer[0] = simulation->bodies[ioffset];
        buffer[1] = simulation->bodies[ioffset+1];
        buffer[2] =  simulation->bodies[ioffset+2];
        
        //write body as bin x,y,z
        fwrite(buffer, sizeof(buffer), 1, f);
    }

    fclose(f);
    return STATUS_OK;
}

