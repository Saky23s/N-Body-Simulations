
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

//Internal helpers
int simulation_allocate_memory(Simulation* simulation);
int save_values_bin(Simulation* simulation, int filenumber);
int save_values_csv(Simulation* simulation, int filenumber);

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
        simulation->n = size / (sizeof(real) * 8); 

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

    //Set time to 0
    simulation-> tnow = 0.0;
   
    //Schedule first output for now
    simulation->tout =  simulation->tnow;
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

    simulation->masses = (realptr) malloc (simulation->n * sizeof(simulation->masses[0]));
    simulation->positions = (realptr) malloc (simulation->n * 3 * sizeof(simulation->positions[0]));
    simulation->velocity = (realptr) malloc (simulation->n * 3 * sizeof(simulation->velocity[0]));
    simulation->acceleration = (realptr) malloc (simulation->n * 3 * sizeof(simulation->acceleration[0]));
    
    if(simulation->masses == NULL || simulation->positions == NULL || simulation->velocity == NULL || simulation->acceleration == NULL)
    {
        return STATUS_ERROR;
    }

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
        int ioffset = i*3;
        if(fprintf(f, "%lf,%lf,%lf\n", simulation->positions[ioffset], simulation->positions[ioffset+1], simulation->positions[ioffset+2]) < 0)
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