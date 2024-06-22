#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include "../inc/simulation.h"
#include "aux.c"

struct _Simulation
{   
    //Bodies variables
    double* masses;
    double* positions;
    double* velocity;
    int n;

    //Variables needed for internals of runge-kutta
    double* k1_position;
    double* k1_velocity;

    double* k2_position;
    double* k2_velocity;

    double* k3_position;
    double* k3_velocity;

    double* k4_position;
    double* k4_velocity;
    
    double* holder_position;
    double* holder_velocity;
    double* block_holder;
} _Simulation;



//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int rk4(Simulation* simulation); 
int save_values_csv(Simulation* simulation, char* filename); 
int save_values_bin(Simulation* simulation, char* filename);
int calculate_acceleration(Simulation* simulation, double*k_position, double* k_velocity);

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

            fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*f\n", &simulation->positions[joffset], &simulation->positions[joffset+1], &simulation->positions[joffset+2], &simulation->masses[j], &simulation->velocity[joffset], &simulation->velocity[joffset+1], &(simulation->velocity[joffset+2]));
            
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
            fread(buffer,sizeof(buffer),1,f);
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

    simulation->k1_position = (double*) malloc (simulation->n * 3 * sizeof(simulation->k1_position[0]));
    simulation->k1_velocity = (double*) malloc (simulation->n * 3 * sizeof(simulation->k1_velocity[0]));

    simulation->k2_position = (double*) malloc (simulation->n * 3 * sizeof(simulation->k2_position[0]));
    simulation->k2_velocity = (double*) malloc (simulation->n * 3 * sizeof(simulation->k2_velocity[0]));

    simulation->k3_position = (double*) malloc (simulation->n * 3 * sizeof(simulation->k3_position[0]));
    simulation->k3_velocity = (double*) malloc (simulation->n * 3 * sizeof(simulation->k3_velocity[0]));

    simulation->k4_position = (double*) malloc (simulation->n * 3 * sizeof(simulation->k4_position[0]));
    simulation->k4_velocity = (double*) malloc (simulation->n * 3 * sizeof(simulation->k4_velocity[0]));

    simulation->holder_position = (double*) malloc (simulation->n * 3 * sizeof(simulation->holder_position[0]));
    simulation->holder_velocity = (double*) malloc (simulation->n * 3 * sizeof(simulation->holder_velocity[0]));

    simulation->block_holder = (double*) malloc (3 * simulation->n * ceil(simulation->n/32.0) * ceil(simulation->n/32.0) * sizeof(simulation->block_holder[0]));

    if(simulation->masses == NULL || simulation->block_holder == NULL
        || simulation->positions == NULL || simulation->velocity == NULL
        || simulation->k1_position == NULL || simulation->k1_velocity == NULL 
        || simulation->k2_position == NULL || simulation->k2_velocity == NULL 
        || simulation->k3_position == NULL || simulation->k3_velocity == NULL 
        || simulation->k4_position == NULL || simulation->k4_velocity == NULL 
        || simulation->holder_position == NULL || simulation->holder_velocity == NULL)
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

    free(simulation->k1_position);
    free(simulation->k1_velocity);

    free(simulation->k2_position);
    free(simulation->k2_velocity);

    free(simulation->k3_position);
    free(simulation->k3_velocity);

    free(simulation->k4_position);
    free(simulation->k4_velocity);

    free(simulation->holder_position);
    free(simulation->holder_velocity);
    free(simulation->block_holder);

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
        fprintf(f, "%lf,%lf,%lf\n", simulation->positions[ioffset], simulation->positions[ioffset+1], simulation->positions[ioffset+2]);
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
        fwrite(buffer, sizeof(buffer), 1, f);
    }

    fclose(f);
    return STATUS_OK;
}

int calculate_acceleration(Simulation* simulation, double*k_position, double* k_velocity)
/**
 * Funtion to calculate the velocity and acceleration of the bodies using the current positions and velocities
 * @param simulation(Simulation*): a pointer to the simulation object we are simulation, in the holder variable the information must be stored as an array of values order as x1,y1,z1,vx1,vz1,vz1,x2,y2,z2,vx2,vz2,vz2...xn,yn,zn,vxn,vzn,vzn
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
**/
{   
    //Error checking
    if(simulation == NULL || k_position == NULL || k_velocity == NULL)
        return STATUS_ERROR;

    //Init values of k as 0
    for(int i = 0; i < simulation->n * 3; i++)
    {
        k_velocity[i] = 0.0;
    }

    //For all of the bodies, in parallel
    #pragma omp parallel for
    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 3;
        k_position[ioffset] = dt * simulation->holder_velocity[ioffset]; //vx
        k_position[ioffset+1] = dt * simulation->holder_velocity[ioffset+1]; //vy
        k_position[ioffset+2] = dt * simulation->holder_velocity[ioffset+2]; //vz

        //For all other bodies
        for(int j = 0; j < simulation->n; j++)
        {   
            //i and j cant be the same body
            if(i==j)
                continue;

            int joffset = j * 3;
            double dx = simulation->holder_position[joffset] - simulation->holder_position[ioffset]; //rx body 2 - rx body 1
            double dy = simulation->holder_position[joffset+1] - simulation->holder_position[ioffset+1]; //ry body 2 - ry body 1
            double dz = simulation->holder_position[joffset+2] - simulation->holder_position[ioffset+2]; //rz body 2 - rz body 1
            
            double r = dx * dx + dy * dy + dz * dz + softening * softening; //distance magnitud with some softening
            double h = dt * (G * simulation->masses[j] / pow(r, 1.5)); //Acceleration formula

            k_velocity[ioffset] += h * dx; //Acceleration formula for x
            k_velocity[ioffset+1] += h * dy; //Acceleration formula for y
            k_velocity[ioffset+2] += h * dz; //Acceleration formula for z
        }
    }
    return STATUS_OK;
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
    for(int i = 0; i < simulation->n*3; i++)
    {
        simulation->holder_position[i] = simulation->positions[i];
        simulation->holder_velocity[i] = simulation->velocity[i];
    }
    
    //Calculate k1
    if(calculate_acceleration(simulation, simulation->k1_position, simulation->k1_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+0.5*k1 to be able to calculate k2
    for(int i = 0; i < simulation->n*3; i++)
    {   
        simulation->holder_position[i] = simulation->positions[i] + simulation->k1_position[i] * 0.5;
        simulation->holder_velocity[i] = simulation->velocity[i] + simulation->k1_velocity[i] * 0.5;
    }

    //Calculate k2
    if(calculate_acceleration(simulation, simulation->k2_position, simulation->k2_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+0.5*k2 to be able to calculate k3
    for(int i = 0; i < simulation->n*3; i++)
    {
        simulation->holder_position[i] = simulation->positions[i] + simulation->k2_position[i] * 0.5;
        simulation->holder_velocity[i] = simulation->velocity[i] + simulation->k2_velocity[i] * 0.5;
    }

    //Calculate k3
    if(calculate_acceleration(simulation, simulation->k3_position, simulation->k3_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+*k3 to be able to calculate k3
    for(int i = 0; i < simulation->n*3; i++)
    {
        simulation->holder_position[i] = simulation->positions[i] + simulation->k3_position[i];
        simulation->holder_velocity[i] = simulation->velocity[i] + simulation->k3_velocity[i];
    }

    //Calculate k4
    if(calculate_acceleration(simulation, simulation->k4_position, simulation->k4_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Update simulation value to simulation.bodies + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)
    for(int i = 0; i < simulation->n*3; i++)
    {
        simulation->positions[i] = simulation->positions[i] + ((simulation->k1_position[i] + 2.0*simulation->k2_position[i] + 2.0*simulation->k3_position[i] + simulation->k4_position[i]) / 6.0);
        simulation->velocity[i] = simulation->velocity[i] + ((simulation->k1_velocity[i] + 2.0*simulation->k2_velocity[i] + 2.0*simulation->k3_velocity[i] + simulation->k4_velocity[i]) / 6.0);

    }

    return STATUS_OK;
}