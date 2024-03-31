#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<unistd.h>
#include <time.h>
#include <string.h>
#include "../inc/simulation.h"


#define BUF_SIZE 65536

//Internal helpers
int count_lines_csv(FILE* f);
void save_values_csv(Simulation* simulation, char* filename);
void rk4(Simulation* simulation);
double* calculate_acceleration(Simulation* simulation, double*values);
int get_extention_type(const char *filename);
void save_values_bin(Simulation* simulation, char* filename);

//Helper macrowords
#ifndef NO_EXT
  #define NO_EXT -1
  #define EXT_CSV 0
  #define EXT_BIN 1
#endif

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

Simulation* load_bodies(char* filepath)
/**
 * This funtion creates a new Simulation and fills it using the starting values from a file
 * @param filepath (char*):  a path to the file with the starting data, must be csv or bin file
 * @return simulation (Simulation*): a pointer to the new Simulation filled with the data in filepath
 */
{   
    //Allocate memory for the Simulation object itself
    Simulation* simulation = (Simulation*) malloc(sizeof(Simulation));
    
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

void run_simulation(Simulation* simulation, double T)
/**
 * Funtion that will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)
 *
 * This funtion will calculate the positions of the bodies every in timesteps of 'dt'using the runge-kutta method
 * and store them in data/ as csv files every 'speed' seconds
 * 
 * @param simulation (Simulation*) pointer to the simulation object with the initial values       
 * @param T (float): Internal ending time of the simulation
**/
{   
    //Calculate the number of steps we will have to take to get to T
    long int steps = T / dt;
    //Calculate the number of timesteps we must do before saving the data
    long int save_step = speed / dt;
    //Internal variables to keep track of csv files written
    long int file_number = 1;
    //Allocate space for the filename
    char *filename = (char*)malloc(256 * sizeof(char));

    //Internal variables to measure time 
    double clock_time = 0.0;
    clock_t start, final, cycles;

    //Measure starting clock time
    start = clock();
    if(start == (clock_t)-1)
    {
      return;
    }

    //Run simulation
    for(long int step = 1; step <= steps; step++)
    {
        //Integrate next step using runge-kutta
        rk4(simulation);
        
        //Save data if we must
        if(step % save_step == 0)
        {   
            FILE* f = NULL;
            
            sprintf(filename, "../Graphics/data/%ld.bin", file_number);
            save_values_bin(simulation, filename);
            
            file_number++;
        }

        //Print fancy progress 
        clock_time += dt;
        printf("\rIntegrating: step = %ld / %ld | simulation time %lf", step, steps, clock_time);
	    fflush(stdout);
    }

    //Measure final clock time
    final = clock();
    if(final == (clock_t)-1)
    {
      return;
    }

    //Calculate how long the simulation took
    cycles = final - start;
    printf("\nSimulation completed in %lf seconds\n", (cycles/(double)CLOCKS_PER_SEC));
    free(filename);

}

void rk4(Simulation* simulation)
/**
 * This funtion will calculate the next values of the simulation using the runge-kutta method
 * 
 * @param simulation (Simulation*): a pointer to the simulation
 **/
{   

    //Correctly set up holder
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->holder[i] = simulation->bodies[i];
    }
    //Calculate k1
    calculate_acceleration(simulation, simulation->k1);

    //Calculate simulation.bodies+0.5*k1 to be able to calculate k2
    for(int i = 0; i < simulation->n*6; i++)
    {   
        simulation->holder[i] = simulation->bodies[i] + simulation->k1[i] * 0.5;
    }

    //Calculate k2
    calculate_acceleration(simulation, simulation->k2);

    //Calculate simulation.bodies+0.5*k2 to be able to calculate k3
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->holder[i] = simulation->bodies[i] + simulation->k2[i] * 0.5;
    }

    //Calculate k3
    calculate_acceleration(simulation, simulation->k3);

    //Calculate simulation.bodies+*k3 to be able to calculate k3
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->holder[i] = simulation->bodies[i] + simulation->k3[i];
    }

    //Calculate k4
    calculate_acceleration(simulation, simulation->k4);

    //Update simulation value to simulation.bodies + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)
    for(int i = 0; i < simulation->n*6; i++)
    {
        simulation->bodies[i] = simulation->bodies[i] + ((simulation->k1[i] + 2.0*simulation->k2[i] + 2.0*simulation->k3[i] + simulation->k4[i]) / 6.0);
    }

    return;
}

double* calculate_acceleration(Simulation* simulation, double*k)
/**
 * Funtion to calculate the velocity and acceleration of the bodies using the current positions and velocities
 * @param simulation(Simulation*): a pointer to the simulation object we are simulation, in the holder variable the information must be stored as an array of values order as x1,y1,z1,vx1,vz1,vz1,x2,y2,z2,vx2,vz2,vz2...xn,yn,zn,vxn,vzn,vzn
 * @return k (double*): pointer in which to save the results, the result will be an array of values order as vx1,vy1,vz1,ax1,ay1,az1,vx2,vy2,vz2,ax2,ay2,az2...vxn,vyn,vzn,axn,ayn,azn
**/
{   
    //Error checking
    if(simulation == NULL || k == NULL)
        return NULL;

    //Init values of k as 0
    for(int i = 0; i < simulation->n * 6; i++)
    {
        k[i] = 0.0;
    }

    //For all of the bodies
    for(int i = 0; i < simulation->n; i++)
    {
        int ioffset = i * 6;
        k[ioffset] = dt * simulation->holder[ioffset+3]; //vx
        k[ioffset+1] = dt * simulation->holder[ioffset+4]; //vy
        k[ioffset+2] = dt * simulation->holder[ioffset+5]; //vz
        //For all other bodies
        for(int j = 0; j < simulation->n; j++)
        {   
            //i and j cant be the same body
            if(i==j)
                continue;

            int joffset = j * 6;
            double dx = simulation->holder[joffset] - simulation->holder[ioffset]; //rx body 2 - rx body 1
            double dy = simulation->holder[joffset+1] - simulation->holder[ioffset+1]; //ry body 2 - ry body 1
            double dz = simulation->holder[joffset+2] - simulation->holder[ioffset+2]; //rz body 2 - rz body 1
            
            double r = sqrt(pow(dx, 2.0) + pow(dy, 2.0) + pow(dz, 2.0) + pow(softening, 2.0)); //distance magnitud with some softening
            

            k[ioffset+3] += dt * (G * simulation->masses[j] / pow(r,3.0)) * dx; //Acceleration formula for x
            k[ioffset+4] += dt * (G * simulation->masses[j] / pow(r,3.0)) * dy; //Acceleration formula for y
            k[ioffset+5] += dt * (G * simulation->masses[j] / pow(r,3.0)) * dz; //Acceleration formula for z
        }
    }
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

void save_values_csv(Simulation* simulation, char* filename)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a csv
 * @param simulation (Simulation*):  a pointer to the simulation being stored
 * @param file (char*) the filepath in which the data is going to be stored as csv
 */
{   
    //Error checking
    if(simulation == NULL || filename == NULL)
        return;

    //Open file
    FILE* f = fopen(filename, "w");
    if(f == NULL)
        return;

    //For all n bodies
    for(int i = 0; i < simulation->n; i++)
    {      
        //Print body as csv x,y,z
        int ioffset = i*6;
        fprintf(f, "%lf,%lf,%lf\n", simulation->bodies[ioffset], simulation->bodies[ioffset+1], simulation->bodies[ioffset+2]);
    }

    fclose(f);
}

void save_values_bin(Simulation* simulation, char* filename)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a bin
 * @param simulation (Simulation*):  a pointer to the simulation being stored
 * @param file (char*) the filepath in which the data is going to be stored as bin
 */
{   
    //Error checking
    if(simulation == NULL || filename == NULL)
        return;

    //Open file
    FILE* f = fopen(filename, "wb");
    if(f == NULL)
        return;

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
}

int get_extention_type(const char *filename) 
/**
 * This funtion will check if a filename is csv, bin or other extention type
 * @param filename (char *):  the filename we are checking
 * @param file_type(int): 
 *                          -1 (NO_EXT) extention not valid
 *                           0 (EXT_CSV) csv file
 *                           1 (EXT_BIN) bin file
 */
{
    const char *dot = strrchr(filename, '.');
    //No extention
    if(!dot || dot == filename) return NO_EXT;

    //If extention is csv
    if(strcmp(dot + 1, "csv") == 0) return EXT_CSV;

    //If extention is bin
    if(strcmp(dot + 1, "bin") == 0) return EXT_BIN;

    //Extention not recognised
    return NO_EXT;
}