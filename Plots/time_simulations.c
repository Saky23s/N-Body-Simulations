#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../Paralelo_Ingenuo/inc/simulation_OpenMP.h"

double random_num(double min, double max);

int main(int argc, char **argv)
/**
 * Small code to generete a log file with the times
 * it takes to simulation to execute a 100 second simulation 
 * with diferent number of bodies
 * @param Max_N (long): The maximum size of N to test
 * @param steps (int): The increment between timing
 * @return 'times.log' will have all the results stored at the end
 * of the excution, in format:
 * 
 * n t
 * 
 * where N is the number of bodies that simulation had
 * and t is the time it took to excute
*/
{
    if(argc != 3)
    {
        printf("Usage: %s <Max_N> <Steps>\n", argv[0]);
        return -1;
    }
    long max_n = atol(argv[1]);
    int step = atoi(argv[2]);

    FILE* output_file = fopen("times.log", "w");
    if(output_file == NULL)
        return -1;

    for(long n = 1; n <= max_n; n += step)
    {   
        //Create an starting position for N bodies
        FILE* position_file = fopen("../Starting_Configurations/bin_files/random.bin", "wb");
        if(position_file == NULL)
        {
            printf("Error opening output file\n");
            return -1;
        }

        double values[8];
        for(int i = 0; i < n; i++)
        {
            values[0] = random_num(-1.0, 1.0);  //x
            values[1] = random_num(-1.0, 1.0);  //y
            values[2] = random_num(-1.0, 1.0);  //z
            values[3] = 1.0;                    //mass
            values[4] = random_num(-1.0, 1.0);  //vx
            values[5] = random_num(-1.0, 1.0);  //vy
            values[6] = random_num(-1.0, 1.0);  //vz
            values[7] = values[3];              //radius

            fwrite(values, sizeof(values), 1, position_file);
        }
        fclose(position_file);
        
        //Create a simulation for that file
        Simulation *simulation = load_bodies("../Starting_Configurations/bin_files/random.bin");    
        if(simulation == NULL)
        {   
            printf("Error while loading simulation\n");
            return -1;
        }
        
        //Run simulation for 100 seconds
        double t = run_simulation(simulation, 100.0);
        
        //Store time
        fprintf(output_file, "%ld %lf\n", n, t);

        //Free memory
        free(simulation);
    }
    fclose(output_file);
}

double random_num(double min, double max) 
{   
    return min + (max + 1.0)*rand()/(RAND_MAX+1.0);
    
}