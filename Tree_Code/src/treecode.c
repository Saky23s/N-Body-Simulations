/** 
 * @file treecode.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * Modifies the original work of Joshua E. Barnes to remove features
 * that are not required for this investigation and make the I-O 
 * system work with our existing framework   
 * 
 * This document combines the I-O functions as well as the main functions for the algorthim
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../../Aux/aux.c"
#define global                                  // don't default to extern
#include "../inc/treedefs.h"
#include <sys/types.h>
#include <sys/time.h>
#include <string.h>

//Internal helpers
local int treeforce(void);
local int leapfrog(void);
int load_bodies(char* filename);
int save_values_bin(int file_number);
int save_values_csv(int file_number);

//body pointer variable
local bodyptr bodytab;


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
    //Internal variable to keep track of csv files written
    int file_number = 1;

    //Read initial data
    if(load_bodies(filename) == STATUS_ERROR)
        return STATUS_ERROR;                     

    //Set starting root w/ unit cube
    rsize = 1.0;                            
    
    //Calculate the number of steps we will have to take to get to T
    int steps = T / dt; 
    //Calculate the number of timesteps we must do before saving the data         
    int save_step = speed / dt;

    printf("Simulating secuentially %d bodies using treecode method\n", nbody);

    //Internal variables to measure time 
    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );
    
    treeforce();
    
    //Run simulation
    for (int step = 0; step < steps; step++)
    {
        //Integrate next step using leapfrog
        if(leapfrog() == STATUS_ERROR)
            return STATUS_ERROR;  
        
        //Save data if we must
        if(step % save_step == 0)
        {
            if(save_values_bin(file_number) == STATUS_ERROR)
                return STATUS_ERROR;
            
            file_number++;
        }
        //Print fancy progress 
        printf("\rIntegrating: step = %d / %d", step, steps);
	    fflush(stdout);
    }

    //Calculate how long the simulation took
    gettimeofday ( &t_end, NULL );
    printf("\nSimulation completed in %lf seconds\n",  WALLTIME(t_end) - WALLTIME(t_start));

    freetree(bodytab);

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

    //Loop over all the bodies and set update forces to true
    for (p = bodytab; p < bodytab+nbody; p++)   
        Update(p) = TRUE;                       
    
    //Construct tree structure
    if(maketree(bodytab, nbody) == STATUS_ERROR)
        return STATUS_ERROR;

    //Calculate the forces
    if(gravcalc() == STATUS_ERROR)
        return STATUS_ERROR;                               

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
        //Half step velocity
        //  vn+1/2 = vn + 1/2dt * a(rn)
        ADDMULVS(Vel(p), Acc(p), 0.5 * dt);

        //Use that velocity to full step the position
        //  rn+1 = rn + dt*vn+1/2
        ADDMULVS(Pos(p), Vel(p), dt);
    }
    
    //Calculate the accelerations with half step velocity and full step position
    if (treeforce() == STATUS_ERROR)
        return STATUS_ERROR; 

    for (p = bodytab; p < bodytab+nbody; p++) 
    { 
        //Half step the velocity again (making a full step)
        //  vn+1 = vn+1/2 + 1/2dt * a(rn+1)
        ADDMULVS(Vel(p), Acc(p), 0.5 * dt);
    }

    return STATUS_OK;
}

int load_bodies(char* filename)
/**
 * This funtion creates uses the starting values from a file to load the N bodies
 * @param filepath (char*):  a path to the file with the starting data, must be csv or bin file
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    bodyptr p;
    FILE* f = NULL;

    //Error checking
    if(filename == NULL)
        return STATUS_ERROR;
    

    int extention_type = get_extention_type(filename);
    if(extention_type == EXT_CSV)
    {   
        //Open file
        f = fopen(filename, "r");
        if(f == NULL)
            return STATUS_ERROR;
        
        //Get the number of bodies by the number of lines minus the header
        nbody = count_lines_csv(f) - 1;
        if(nbody <= 0)
            return STATUS_ERROR;
        
        //Memory allocation for the arrays
        bodytab = (bodyptr) calloc(nbody * sizeof(body), 1);
        if(bodytab == NULL)
            return STATUS_ERROR;
        //go back to the begining of file
        rewind(f);

        p = bodytab; 

        //For the number of bodies + header
        for(int i = 0; i < nbody + 1; i++)
        {     
            //read header
            if(i == 0)
            {   
                //skip header line
                fscanf(f, "%*[^\n]\n");
                continue;
            }

            //Read body
            if(fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*f\n", &(Pos(p)[0]), &(Pos(p)[1]), &(Pos(p)[2]), &Mass(p), &(Vel(p)[0]), &(Vel(p)[1]), &(Vel(p)[2])) == EOF)
            {
                printf("Error reading %s\n", filename);
                return STATUS_ERROR;
            }
            Type(p) = BODY;
            p++;
        }
        fclose(f);   
    }                       
    else if (extention_type == EXT_BIN)
    {
        //Read as binary
        FILE* f = fopen(filename, "rb");
        if(f == NULL)
            return STATUS_ERROR;
        

        //Get file size
        fseek(f, 0, SEEK_END); 
        long size = ftell(f); 
        fseek(f, 0, SEEK_SET);

        //The number of bodies is the size of the file / size of each body
        nbody = size / (sizeof(double) * 8); 

        //Memory allocation for the arrays
        bodytab = (bodyptr) calloc(nbody * sizeof(body), 1);
        if(bodytab == NULL)
            return STATUS_ERROR;
        
        //Buffer for one body
        double buffer[8];
        
        //Read the whole file
        for (p = bodytab; p < bodytab+nbody; p++)
        {   
            if(fread(buffer,sizeof(buffer),1,f) == 0)
                return STATUS_ERROR;
                
            Pos(p)[0] = buffer[0];  //x
            Pos(p)[1] = buffer[1];  //y
            Pos(p)[2] = buffer[2];  //z
            Mass(p) = buffer[3];    //mass
            Vel(p)[0] = buffer[4];  //vx
            Vel(p)[1] = buffer[5];  //vy
            Vel(p)[2] = buffer[6];  //vz

            //Buffer[7] is radius, currently useless for data, only useful for graphics
            Type(p) = BODY;
        }
        fclose(f);
    }
    //File type not recognized
    else
    {
        return STATUS_ERROR;
    }
    return STATUS_OK;
}

int save_values_bin(int file_number)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a bin
 * The file will be stored as /dev/shm/data/FILE_NUMBER.bin
 * @param file_number (int) the file number to be used 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    char filename[256];
    bodyptr p;

    //Construct output name
    snprintf(filename, 256, "/dev/shm/data/%d.bin", file_number);
    
    //Open file
    FILE* f = fopen(filename, "wb");
    if(f == NULL)
        return STATUS_ERROR;
    
    //Print all bodies
    for (p = bodytab; p < bodytab+nbody; p++)
    {
        if (fwrite((void *)  Pos(p), sizeof(real), NDIM, f) != NDIM)
        {
            printf("out_vector: fwrite failed\n");
            return STATUS_ERROR;
        }
    }

    //Close up output file 
    fclose(f);
    return STATUS_OK;
}

int save_values_csv(int file_number)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a csv
 * The file will be stored as /dev/shm/data/FILE_NUMBER.csv
 * @param file_number (int) the filenumber to be used 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{   
    char filename[256];
    bodyptr p;

    //Construct output name
    snprintf(filename, 256, "/dev/shm/data/%d.csv", file_number);
    
    //Open file
    FILE* f = fopen(filename, "w");
    if(f == NULL)
        return STATUS_ERROR;

    //For all n bodies
    for (p = bodytab; p < bodytab+nbody; p++)
    {      
        //Print body as csv x,y,z
        if(fprintf(f, "%lf,%lf,%lf\n", Pos(p)[0], Pos(p)[1], Pos(p)[2]) < 0)
            return STATUS_ERROR;
    }

    fclose(f);
    return STATUS_OK;
}




