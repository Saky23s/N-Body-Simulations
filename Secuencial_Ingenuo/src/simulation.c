#include <stdio.h>
#include <stdlib.h>
#include "../inc/simulation.h"

#define BUF_SIZE 65536

//Little internal helper to get the number of bodies if they are given with an csv file
int count_lines_csv(FILE* f);

Simulation* load_bodies(char* filepath)
/**
 * This funtion creates a new Simulation and fills it using the starting values from a file
 * @param filepath (char*):  a path to the file with the starting data (CURRENTLY ONLY CSV FILES)
 * @return simulation (Simulation*): a pointer to the new Simulation filled with the data in filepath
 * 
 * 
 * @todo TODOOOOOOOO ADD SUPPORT FOR BINARY FILES
 */
{   
    //Allocate memory for the Simulation object itself
    Simulation* simulation = (Simulation*) malloc(sizeof(Simulation));
    
    //Error checking
    if(filepath == NULL)
    {
        return NULL;
    }
    
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
    simulation->bodies = (double*) malloc ((simulation->n * 6)*sizeof(simulation->bodies[0]));
    simulation->masses = (double*) malloc (simulation->n * sizeof(simulation->masses[0]));
    if(simulation->bodies == NULL || simulation->masses == NULL)
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

    return simulation;
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

int count_lines_csv(FILE* f)
/**
 * This returns the number of lines in a text file, is O(N) but its the only way
 * @param file (FILE*) an openned file which lines are going to be read
 * @return n_lines (int) the number of lines in the file
 */
{   
    //Buffer to read all data too
    char buffer[BUF_SIZE];
    int n_lines = 0;
    //Til we read the whole file
    for(;;)
    {   
        //Load a lot of characters 
        size_t res = fread(buffer, 1, BUF_SIZE, f);
        if (ferror(f))
            return -1;

        //Iterate the characters and count the \n
        int i;
        for(i = 0; i < res; i++)
            if (buffer[i] == '\n')
                n_lines++;

        //If we reached the end of the file stop reading
        if (feof(f))
            break;
        //If we havent reached the end keep going
    }

    return n_lines;
}