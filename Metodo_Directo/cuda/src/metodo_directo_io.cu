/** 
 * @file metodo_directo_io.c
 * @author Santiago Salas santiago.salas@estudiante.uam.es
 * 
 * This file is in charge of doing all of the input-output operations
 * this means 
 * 
 *  - Reading starting configurations and loading it to the simulation
 *  - Allocating such simulation dinamically for the stating configuration
 *  - Outputing ever 'speed' seconds the current position of the N bodies as binary or csv files 
 *  - Free all the memory allocated
 */

#include "../inc/medoto_directo_defs.h"
#include "../../../Aux/aux.c"

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

//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int save_values_bin(Simulation* simulation, int filenumber);
int save_values_csv(Simulation* simulation, int filenumber);
int calculate_kernel_size(Simulation* simulation);

//Internal variables to control file output
#define FILENAME_MAX_SIZE 256

int output(Simulation* simulation, int* filenumber)
/**
 * 
 * This funtion will control the outputs.
 * 
 * It will also check if its time to save the positions of the bodies to a file, in which case it will and
 * it will schedule the next output 
 * 
 * @param simulation (Simulation*):  a pointer to the simulation being outputed
 * @param filenumber (*int): ponter to the number of the file to use in the output, if a writte is made it will automatically be incremented by 1
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    real teff;

    //Anticipate slightly...
    teff = simulation->tnow + dt/8;

    if (teff >= simulation->tout)
    {   
        //Increment filenumber by one
        (*filenumber) += 1;

        //Pass the positions back from gpu
        cudaMemcpy( simulation->positions, simulation->d_positions, simulation->n * 3 * sizeof(simulation->d_positions[0]), cudaMemcpyDeviceToHost);
                
        //Save
        if(save_values_bin(simulation, *filenumber) == STATUS_ERROR)
            return STATUS_ERROR;

        //Schedule next output
        simulation->tout += speed;
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

        //Calculate kernel sizes
        if (calculate_kernel_size(simulation) == STATUS_ERROR)
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
            //read header
            if(i == 0)
            {   
                //skip header line
                fscanf(f, "%*[^\n]\n", NULL);
                continue;
            }

            if(fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*f\n", &simulation->positions[j], &simulation->positions[j+simulation->n], &simulation->positions[j+2*simulation->n], &simulation->masses[j], &simulation->velocity[j], &simulation->velocity[j+simulation->n], &(simulation->velocity[j+2*simulation->n])) == EOF)
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
        simulation->n = size / (sizeof(real) * 8); 

        //Calculate kernel sizes
        if (calculate_kernel_size(simulation) == STATUS_ERROR)
        {
            return NULL;
        }

        //Memory allocation for the arrays
        if(simulation_allocate_memory(simulation) != STATUS_OK)
        {
            return NULL;
        }
        
        
        //Buffer for one body
        real buffer[8];
        //Read the whole file
        for (int i = 0; i < simulation->n; i++)
        {   
            if(fread(buffer,sizeof(buffer),1,f) == 0)
                return NULL;
                
            simulation->positions[i] = buffer[0];     //x
            simulation->positions[i + simulation->n] = buffer[1];   //y
            simulation->positions[i + 2 * simulation->n] = buffer[2];   //z
            simulation->masses[i] = buffer[3];              //mass
            simulation->velocity[i] = buffer[4];      //vx
            simulation->velocity[i + simulation->n] = buffer[5];    //vy
            simulation->velocity[i + 2 * simulation->n] = buffer[6];    //vz

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
    cudaMemcpy( simulation->d_positions,  simulation->positions, 3 * simulation->n * sizeof(simulation->d_positions[0]),cudaMemcpyHostToDevice);
    cudaMemcpy( simulation->d_velocity,  simulation->velocity, 3 * simulation->n * sizeof(simulation->d_velocity[0]),cudaMemcpyHostToDevice);

    //Set time to 0
    simulation-> tnow = 0.0;
   
    //Schedule first output for now
    simulation->tout =  simulation->tnow;

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

    simulation->masses = (realptr) malloc (simulation->n * sizeof(simulation->masses[0]));
    simulation->positions = (realptr) malloc (3 * simulation->n * sizeof(simulation->positions[0]));
    simulation->velocity = (realptr) malloc (3 * simulation->n * sizeof(simulation->velocity[0]));
    if(simulation->masses == NULL || simulation->positions == NULL || simulation->velocity == NULL)
    {
        return STATUS_ERROR;
    }

    cudaError_t status;

    //Cuda mallocs
    status = cudaMalloc(&simulation->d_masses, simulation->n * sizeof(simulation->d_masses[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_positions, 3 * simulation->n * sizeof(simulation->d_positions[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_acceleration,3 * simulation->n * sizeof(simulation->d_acceleration[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_velocity,3 * simulation->n * sizeof(simulation->d_velocity[0]));
    cudaErrorCheck(status);

    status = cudaMalloc(&simulation->d_block_holder, simulation->n  * simulation->gridDimsGrav.y * 3 * sizeof(simulation->d_block_holder[0]));
    cudaErrorCheck(status);

    return STATUS_OK;
}


int save_values_csv(Simulation* simulation, int filenumber)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a csv
 * The output will be stored in "/dev/shm/data/{filename}.csv" where filename is the current number of outputs done
 * 
 * the data is stored as N lines each line contains the x,y,z position for one body
 * 
 * @param simulation (Simulation*):  a pointer to the simulation being stored
 * @param filenumber (int): Number of the file to be created
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{   
    char filename[256];
    
    //Error checking
    if(simulation == NULL)
        return STATUS_ERROR;

    //Construct output name
    snprintf(filename, 256, "/dev/shm/data/%d.csv", filenumber);

    //Open file
    FILE* f = fopen(filename, "w");
    if(f == NULL)
        return STATUS_ERROR;

    //For all n bodies
    for(int i = 0; i < simulation->n; i++)
    {      
        //Print body as csv x,y,z
        if(fprintf(f, "%lf,%lf,%lf\n", simulation->positions[i], simulation->positions[i + simulation->n], simulation->positions[i + 2*simulation->n]) < 0)
            return STATUS_ERROR;
    }

    fclose(f);
    return STATUS_OK;
}

int save_values_bin(Simulation* simulation, int filenumber)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a binary file
 * The output will be stored in "/dev/shm/data/{filename}.bin" where filename is the current number of outputs done
 * 
 * the data is stored as x,y,z binary for all of the bodies
 * 
 * @param simulation (Simulation*):  a pointer to the simulation being stored
 * @param filenumber (int): Number of the file to be created
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{   
    char filename[256];
    
    //Error checking
    if(simulation == NULL)
        return STATUS_ERROR;

    //Construct output name
    snprintf(filename, 256, "/dev/shm/data/%d.bin", filenumber);

    //Open file
    FILE* f = fopen(filename, "wb");
    if(f == NULL)
        return STATUS_ERROR;

    real buffer[3];
    
    //For all n bodies
    for(int i = 0; i < simulation->n; i++)
    {      
        buffer[0] = simulation->positions[i];
        buffer[1] = simulation->positions[i + simulation->n];
        buffer[2] =  simulation->positions[i + 2*simulation->n];
        
        //write body as bin x,y,z
        if(fwrite(buffer, sizeof(buffer), 1, f) == 0)
            return STATUS_ERROR;
    }

    fclose(f);
    return STATUS_OK;
}

int calculate_kernel_size(Simulation* simulation)
/**
 * A simple funtion to set kernel sizes for this simulation depending of the size of N
 * @param simulation (Simulation*): a pointer to the simulation
 */
{   
    //Calculate kernel sizes
    unsigned int x = 1;
    unsigned int y = 256;

    simulation->threadBlockDimsGrav = {x, y, 1} ; //256 threads per block
    simulation->gridDimsGrav = { (unsigned int) (simulation->n + x - 1) / x,  (unsigned int) (simulation->n + y - 1) / y, 1 }; 

    x = 256;
    simulation->threadBlockLeap = x; //256 threads per block
    simulation->gridDimsLeap = (unsigned int) (simulation->n + x - 1) / x; 

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

    cudaFree(simulation->d_masses);
    cudaFree(simulation->d_acceleration);
    cudaFree(simulation->d_positions);
    cudaFree(simulation->d_velocity);
    cudaFree(simulation->d_block_holder);

    //Frees the simulation object itself
    free(simulation);
}