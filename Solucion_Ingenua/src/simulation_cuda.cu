#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "../inc/simulation.h"
#include "aux.c"

//Macros to correctly access multidimensional arrays that has been flatten 
#define B(n, size_j, cord, i, j, pointer) pointer[(n * size_j * cord) + (i * size_j) + j]
#define S(size_i, size_j, cord, i, j, pointer) pointer[(size_i * size_j * cord) + (size_j * i) + j]

#define FILENAME_MAX_SIZE 256

/**
 * @struct Simulation
 * @brief Structure with the information for the generation of the N body simulation
 *
 * Structure declaration for the simulation, structured in the form
 * that the data is optimized to minimize cache misses in the CPU
 * and with all the data needed to use the GPU to do the calculations
 */
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
    
    //Cuda variables
    double* d_masses;
    double* d_position;
    double* d_k_velocity;
    dim3 threadBlockDims;
    dim3 gridDims;

} _Simulation;

//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int rk4(Simulation* simulation); 
int save_values_csv(Simulation* simulation, char* filename); 
int save_values_bin(Simulation* simulation, char* filename);
int calculate_kernel_size(Simulation* simulation);
int calculate_acceleration(Simulation* simulation, double*k_position, double* k_velocity);

//Cuda kernels
__global__ void calculate_acceleration_values_block_reduce(double* d_masses, double* d_position, double* d_block_holder, int n, double d_dt, unsigned int number_of_blocks_j);
__device__ void calculate_acceleration_values(double* d_masses, double* d_position, double* sdata, int n, double d_dt);
__device__ void full_block_reduction (double* d_block_holder, double* sdata, int n, unsigned int number_of_blocks_j);


#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/**
 * Function to check errors in CUDA. 
 * 
 * Extracted from StackOverflow 
 * @link https://stackoverflow.com/a/14038590
 */
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

            //Read bodies
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

        //Calculate kernel sizes
        if( calculate_kernel_size(simulation) == STATUS_ERROR )
        {
            free_simulation(simulation);
            return STATUS_ERROR;
        }

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
    
    //Copy masses to cuda memory
    cudaError_t status = cudaMemcpy( simulation->d_masses,  simulation->masses, simulation->n * sizeof(simulation->masses[0]),cudaMemcpyHostToDevice);
    cudaErrorCheck(status);
    

    //Return simulation
    return simulation;
}

int simulation_allocate_memory(Simulation* simulation)
/**
 * Funtion that allocates all of the internal memory needed for the simulation
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

    cudaError_t status;

    //Cuda mallocs
    status = cudaMalloc(&simulation->d_masses, simulation->n * sizeof(simulation->d_masses[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_position, simulation->n * 3 * sizeof(simulation->d_position[0]));
    cudaErrorCheck(status);

    if(simulation->n <= 32)
        status = cudaMalloc(&simulation->d_k_velocity, simulation->n * 3 * sizeof(simulation->d_k_velocity[0]));
    else
        status = cudaMalloc(&simulation->d_k_velocity, simulation->n * simulation->gridDims.x * simulation->gridDims.y * 3 * sizeof(simulation->d_k_velocity[0]));
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
    free(simulation->block_holder);

    cudaFree(simulation->d_masses);
    cudaFree(simulation->d_position);
    cudaFree(simulation->d_k_velocity);

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

    char filename[FILENAME_MAX_SIZE];

    //Internal variables to measure time 
    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    printf("Simulating with CUDA\n");

    //Run simulation
    for(long int step = 1; step <= steps; step++)
    {
        //Integrate next step using runge-kutta
        if(rk4(simulation) == STATUS_ERROR)
            return STATUS_ERROR;
        
        //Save data if we must
        if(step % save_step == 0)
        {   
            if(snprintf(filename, FILENAME_MAX, "../Graphics/data/%ld.bin", file_number) < 0)
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
 * Funtion to calculate the velocity and acceleration of the bodies using the current positions and velocities. It uses a cuda kernel to calculate the values
 * @param simulation(Simulation*): a pointer to the simulation object we are simulating
 * @param k_position (double*): Array to store resulting positions of the N bodies. They are stored as follows x1,y1,z1,x2,y2,z2....xn,yn,zn
 * @param k_velocity (double*): Array to store the resulting velocities of the N bodies. They are stored as follows vx1,vy1,vz1,vx2,vy2,vz2....vxn,vyn,vzn
 * 
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

    //Set up memory for cuda
    cudaMemcpy( simulation->d_position,  simulation->holder_position, simulation->n * 3 * sizeof(simulation->holder_position[0]),cudaMemcpyHostToDevice);
    cudaMemset( simulation->d_k_velocity, 0.0, 3 * simulation->n  * simulation->gridDims.y * sizeof(simulation->d_k_velocity[0]));
    
    //Call cuda
    calculate_acceleration_values_block_reduce<<<simulation->gridDims, simulation->threadBlockDims, 3 * simulation->threadBlockDims.x * simulation->threadBlockDims.y * sizeof(double)>>>(simulation->d_masses, simulation->d_position, simulation->d_k_velocity ,simulation->n, dt, simulation->gridDims.y);
    cudaError_t status = cudaGetLastError();
    cudaErrorCheck(status);

    //Pass results to cpu
    cudaMemcpy( simulation->block_holder, simulation->d_k_velocity, 3 * simulation->n * simulation->gridDims.y * sizeof(simulation->block_holder[0]), cudaMemcpyDeviceToHost);
    
    //Aggregate results
    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 3;
        for(int j = 0; j < simulation->gridDims.y; j++)
        {
            k_velocity[ioffset] += B(simulation->n,simulation->gridDims.y, 0, i, j, simulation->block_holder);
            k_velocity[ioffset + 1] += B(simulation->n,simulation->gridDims.y, 1, i, j, simulation->block_holder);
            k_velocity[ioffset + 2] += B(simulation->n,simulation->gridDims.y, 2, i, j, simulation->block_holder);
        } 
    }
    return STATUS_OK;
}

__device__ void calculate_acceleration_values(double* d_masses, double* d_position, double* sdata, int n, double d_dt)
/**
 * Cuda kernel that calculates the acceleration values that each body suffers from every other body
 * @param d_masses (double*): A cuda array of size n with the masses of each body
 * @param d_position (double*): A cuda array of size 3n with the current position of all of the bodies stored in the following matter x1,y1,z1,x2,y2,z2...xn,yn,zn
 * @param sdata (double*): A shared array in which to store all of the acceleration values
 * @param n (int): the number of bodies
 * @param d_dt (double): the timestep increment
 */
{
    //Get position in the block
    int x = threadIdx.x;
    int y = threadIdx.y;

    //Get universal position
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int ioffset = i * 3;
    int joffset = j * 3;
    
    
    //Calculate pull of one body by other body
    if(i < n && j < n && i != j)
    {   
        double dx = d_position[joffset] - d_position[ioffset]; //rx body 2 - rx body 1
        double dy = d_position[joffset+1] - d_position[ioffset+1]; //ry body 2 - ry body 1
        double dz = d_position[joffset+2] - d_position[ioffset+2]; //rz body 2 - rz body 1
        
        double r = dx * dx + dy * dy + dz * dz + softening * softening; //distance magnitud with some softening
        double h = ((G * d_masses[j]) / (pow(r, 1.5))); //Acceleration formula

        S(blockDim.x, blockDim.y, 0, x, y, sdata) = d_dt * h * dx; //Acceleration formula for x
        S(blockDim.x, blockDim.y, 1, x, y, sdata) = d_dt * h * dy; //Acceleration formula for y
        S(blockDim.x, blockDim.y, 2, x, y, sdata) = d_dt * h * dz; //Acceleration formula for z
    }
    //Fill with 0 the remaining values in the array with 0
    else
    {  
        S(blockDim.x, blockDim.y, 0, x, y, sdata) = 0.0;    //x
        S(blockDim.x, blockDim.y, 1, x, y, sdata) = 0.0;    //y
        S(blockDim.x, blockDim.y, 2, x, y, sdata) = 0.0;    //z
    }
    
}
__global__ void calculate_acceleration_values_block_reduce(double* d_masses, double* d_position, double* d_block_holder, int n, double d_dt, unsigned int number_of_blocks_j)
/**
 * A cuda kernel that calculate the 3n**2 acceleration values and reduce them to an array of size of 3n * gridDim.y. 
 * @param d_masses (double*): A cuda array of size n with the masses of each body
 * @param d_position (double*): A cuda array of size 3n with the current position of all of the bodies stored in the following matter x1,y1,z1,x2,y2,z2...xn,yn,zn
 * @param d_block_holder (double*): An array where the block-reduced results will be stored. It must be of size 3n * gridDim.y
 * @param n (int): the number of bodies
 * @param d_dt (double): the timestep increment
 * @param number_of_blocks_j (unsigned int): The number of blocks in the j dimension. Used for the block reduction
 */
{   
    //Array where all values of this block will be stored
    extern __shared__ double sdata[];

    //Calculate aceleration values
    calculate_acceleration_values(d_masses, d_position, sdata, n, d_dt);
        
    //Reduce all values of this block
    full_block_reduction(d_block_holder, sdata, n, number_of_blocks_j);
}

int calculate_kernel_size(Simulation* simulation)
/**
 * A simple funtion to calculate the most efficient kernel sizes for this simulation depending of the size of N
 * @param simulation (Simulation*): a pointer to the simulation
 */
{
    if(simulation == NULL)
        return STATUS_ERROR;
    
    unsigned int x = 32;
    unsigned int y = 32;

    for(; y <= 1024; x/=2, y*=2)
    {
        if(simulation->n <= y)
        {
            simulation->threadBlockDims = {x, y, 1} ; //1024 threads per block
            simulation->gridDims = { (unsigned int) ceil(simulation->n/(double) x), (unsigned int) ceil( simulation->n/(double) y), 1 }; 
            return STATUS_OK;
        }
    }

    x = 1;
    y = 1024;

    simulation->threadBlockDims = {x, y, 1} ; //1024 threads per block
    simulation->gridDims = { (unsigned int) ceil(simulation->n/(double) x), (unsigned int) ceil( simulation->n/(double) y), 1 }; 

    return STATUS_OK;
}

__device__ void full_block_reduction (double* d_block_holder, double* sdata, int n, unsigned int number_of_blocks_j)
/**
 * A cuda kernel that reduces the acceleration values of this block to one for every body in this block
 * @param d_block_holder (double*): An array where the block-reduced results will be stored. It must be of size 3n * gridDim.y
 * @param sdata (double*): A shared array in which all of the aceleration values of this block are stored
 * @param n (int): the number of bodies
 * @param number_of_blocks_j (unsigned int): The number of blocks in the j dimension.
 */
{
    //Get position in the block
    int x = threadIdx.x;
    int y = threadIdx.y;

    //Get universal position
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.y/2; s>0; s>>=1) 
    {
        if (y < s) 
        {
            S(blockDim.x, blockDim.y, 0, x, y, sdata) += S(blockDim.x, blockDim.y, 0, x, y + s, sdata);
            S(blockDim.x, blockDim.y, 1, x, y, sdata) += S(blockDim.x, blockDim.y, 1, x, y + s, sdata);
            S(blockDim.x, blockDim.y, 2, x, y, sdata) += S(blockDim.x, blockDim.y, 2, x, y + s, sdata);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (y == 0)
    {
        B(n, number_of_blocks_j, 0, i, blockIdx.y, d_block_holder) += S(blockDim.x, blockDim.y, 0, x, 0, sdata);
        B(n, number_of_blocks_j, 1, i, blockIdx.y, d_block_holder) += S(blockDim.x, blockDim.y, 1, x, 0, sdata);
        B(n, number_of_blocks_j, 2, i, blockIdx.y, d_block_holder) += S(blockDim.x, blockDim.y, 2, x, 0, sdata);
    } 
}

