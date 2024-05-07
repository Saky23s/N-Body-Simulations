#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<unistd.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include "../inc/simulation.h"
#include "aux.c"

//MACROS to correctly access multidimensional flattened arrays
#define D(n, cord, i, j, pointer) pointer[(n * n * cord) + (i * n) + j]
#define P(n, cord, i, j) simulation->acceleration[(n * n * cord) + (i * n) + j]

struct _Simulation
{
    double* masses;
    double* positions;
    double* velocity;
    int n;

    //Variables needed for internals
    double* acceleration;
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
    
    //Cuda variables
    double* d_masses;
    double* d_acceleration;
    double* d_holder_position;
} _Simulation;

//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int rk4(Simulation* simulation); 
int save_values_csv(Simulation* simulation, char* filename); 
int save_values_bin(Simulation* simulation, char* filename);
int calculate_acceleration(Simulation* simulation, double*k_position,double*k_velocity);

//Cuda kernel
__global__ void calculate_acceleration_values(double* d_masses, double* d_holder_position, double* d_aceleration, int n, double d_dt);

//Cuda error checking
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
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
    
    //Copy masses to cuda memory
    cudaMemcpy( simulation->d_masses,  simulation->masses, simulation->n * sizeof(simulation->masses[0]),cudaMemcpyHostToDevice);
    
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

    simulation->acceleration = (double*) malloc (3 * simulation->n * simulation->n * sizeof(simulation->acceleration[0]));

    if(simulation->masses == NULL || simulation->acceleration == NULL
        || simulation->positions == NULL || simulation->acceleration == NULL
        || simulation->k1_position == NULL || simulation->k1_velocity == NULL 
        || simulation->k2_position == NULL || simulation->k2_velocity == NULL 
        || simulation->k3_position == NULL || simulation->k3_velocity == NULL 
        || simulation->k4_position == NULL || simulation->k4_velocity == NULL 
        || simulation->holder_position == NULL || simulation->holder_velocity == NULL)
    {
        return STATUS_ERROR;
    }

    cudaError_t status;

    //Cuda mallocs
    status = cudaMalloc(&simulation->d_masses, simulation->n * sizeof(simulation->d_masses[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_acceleration, 3 * simulation->n * simulation->n * sizeof(simulation->d_acceleration[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_holder_position, simulation->n * 3 * sizeof(simulation->d_holder_position[0]));
    cudaErrorCheck(status);

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

    free(simulation->acceleration);

    cudaFree(simulation->d_acceleration);
    cudaFree(simulation->d_masses);
    cudaFree(simulation->d_holder_position);

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
 * @return t (double): Real time that the simulation was running, STATUS_ERROR in case something went wrong
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
    for(int i = 0; i < simulation->n * 3; i++)
    {
        simulation->holder_position[i] = simulation->positions[i];
        simulation->holder_velocity[i] = simulation->velocity[i];
    }
    
    //Calculate k1
    if(calculate_acceleration(simulation, simulation->k1_position, simulation->k1_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+0.5*k1 to be able to calculate k2
    for(int i = 0; i < simulation->n * 3; i++)
    {   
        simulation->holder_position[i] = simulation->positions[i] + simulation->k1_position[i] * 0.5;
        simulation->holder_velocity[i] = simulation->velocity[i] + simulation->k1_velocity[i] * 0.5;
    }

    //Calculate k2
    if(calculate_acceleration(simulation, simulation->k2_position, simulation->k2_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+0.5*k2 to be able to calculate k3
    for(int i = 0; i < simulation->n * 3; i++)
    {
        simulation->holder_position[i] = simulation->positions[i] + simulation->k2_position[i] * 0.5;
        simulation->holder_velocity[i] = simulation->velocity[i] + simulation->k2_velocity[i] * 0.5;
    }

    //Calculate k3
    if(calculate_acceleration(simulation, simulation->k3_position, simulation->k3_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate simulation.bodies+*k3 to be able to calculate k3
    for(int i = 0; i < simulation->n * 3; i++)
    {
        simulation->holder_position[i] = simulation->positions[i] + simulation->k3_position[i];
        simulation->holder_velocity[i] = simulation->velocity[i] + simulation->k3_velocity[i];
    }

    //Calculate k4
    if(calculate_acceleration(simulation, simulation->k4_position, simulation->k4_velocity) == STATUS_ERROR)
        return STATUS_ERROR;

    //Update simulation value to simulation.bodies + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)
    for(int i = 0; i < simulation->n * 3; i++)
    {
        simulation->positions[i] = simulation->positions[i] + ((simulation->k1_position[i] + 2.0*simulation->k2_position[i] + 2.0*simulation->k3_position[i] + simulation->k4_position[i]) / 6.0);
        simulation->velocity[i] = simulation->velocity[i] + ((simulation->k1_velocity[i] + 2.0*simulation->k2_velocity[i] + 2.0*simulation->k3_velocity[i] + simulation->k4_velocity[i]) / 6.0);
    }

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
    if(simulation == NULL || k_position == NULL|| k_velocity == NULL)
        return STATUS_ERROR;

    //Init values of k
    for(int i = 0; i < simulation->n; i++)
    {   
        int ioffset = i * 3;
        k_position[ioffset] = dt * simulation->holder_velocity[ioffset];
        k_position[ioffset+1] = dt * simulation->holder_velocity[ioffset+1];
        k_position[ioffset+2] = dt * simulation->holder_velocity[ioffset+2];
        k_velocity[ioffset] = 0.0;
        k_velocity[ioffset+1] = 0.0;
        k_velocity[ioffset+2] = 0.0;
    }

    //Call cuda
    dim3 threadBlockDims;
    dim3 gridDims;

    //If simulation is smaller (or equal) to one block
    if(simulation->n <= 32)
    {
        threadBlockDims= {(unsigned int) simulation->n , (unsigned int) simulation->n , 1 } ; //up to 1024 threads per block
        gridDims = { 1, 1, 1 } ; 
    }
    //If simulation needs multiple blocks
    else
    {   
        threadBlockDims = {32 , 32 , 1 } ; //1024 threads per block
        gridDims = { (unsigned int) ceil( simulation->n/32.0), (unsigned int) ceil(simulation->n/32.0), 1 } ;  
    }
    
    //Copy position data from host to device.
    cudaMemcpy( simulation->d_holder_position,  simulation->holder_position, simulation->n * 3 * sizeof(simulation->holder_position[0]),cudaMemcpyHostToDevice);

    //Launch kernel
    calculate_acceleration_values<<<gridDims , threadBlockDims >>>(simulation->d_masses,simulation->d_holder_position, simulation->d_acceleration, simulation->n, dt);
    
    //Get errors
    cudaError_t status = cudaGetLastError();
    cudaErrorCheck(status);

    //Copy acceleration values from kernel to host
    cudaMemcpy( simulation->acceleration, simulation->d_acceleration, simulation->n * simulation->n * 3 * sizeof(simulation->acceleration[0]), cudaMemcpyDeviceToHost);
    
    //Reduce acceleration values in CPU using OpenMP
    #pragma omp parallel for
    for(int i = 0; i < simulation->n; i++)
    {   
        int ioffset = i * 3;
        for(int j = 0; j < simulation->n; j++)
        {
            k_velocity[ioffset] += P(simulation->n, 0, i, j);
            k_velocity[ioffset + 1] += P(simulation->n, 1, i, j);
            k_velocity[ioffset + 2] += P(simulation->n, 2, i, j);
        }
    }
    return STATUS_OK;
}

__global__ void calculate_acceleration_values(double* d_masses, double* d_holder_position, double* d_aceleration, int n, double d_dt)
/**
 * Cuda Kernel to calculate the n**2 accelerations that every body suffers from every other body
 * @param d_masses (double*): device memory pointer to all of the body masses
 * @param d_holder_position (double*): device memory pointer with the x,y,z positions of all bodies
 * @param d_aceleration (double*): device memory pointer of 3 * n * n size where all the accelerations will be stored, in i, j we will store the acceleration that i suffers from j gravitational pull
 * @param n (int): number of bodies
 * @param d_dt (double): delta time
**/
{   
    //Get universal position
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //Dont calculate the pull of one body to itself
    if(i == j)
    {
        D(n, 0, i, j, d_aceleration) = 0.0;    //x
        D(n, 1, i, j, d_aceleration) = 0.0;    //y
        D(n, 2, i, j, d_aceleration) = 0.0;    //z
    }
    else if(i < n && j < n)
    {   
        int ioffset = i * 3;
        int joffset = j * 3;

        double dx = d_holder_position[joffset] - d_holder_position[ioffset]; //rx body 2 - rx body 1
        double dy = d_holder_position[joffset+1] - d_holder_position[ioffset+1]; //ry body 2 - ry body 1
        double dz = d_holder_position[joffset+2] - d_holder_position[ioffset+2]; //rz body 2 - rz body 1
        
        double r = pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(softening, 2); //distance magnitud with some softening
        double h = ((G * d_masses[j]) / (pow(r, 1.5))); //Acceleration formula


        D(n, 0, i, j, d_aceleration) = d_dt * h * dx; //Acceleration formula for x
        D(n, 1, i, j, d_aceleration) = d_dt * h * dy; //Acceleration formula for y
        D(n, 2, i, j, d_aceleration) = d_dt * h * dz; //Acceleration formula for z
    }
}

