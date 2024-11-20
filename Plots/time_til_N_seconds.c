#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../Metodo_Directo/simulation.h"

double random_num(double min, double max);



int main(int argc, char **argv)
/**
 * Small code to generete a log file with the times
 * it takes to simulation to execute a 100 second simulation 
 * with diferent number of bodies
 * @param Starting_N (long): The starting size of N to test
 * @param Max_time (long): The maximum time the simulation can take until stopping
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
    if(argc != 4)
    {
        printf("Usage: %s <Starting_N> <Max_N> <Steps>\n", argv[0]);
        return STATUS_ERROR;
    }
    long n = atol(argv[1]);
    double max_time = strtod(argv[2], NULL);
    int step = atoi(argv[3]);

    //Create an empty file
    FILE* output_file = fopen("times.log", "w");
    if(output_file == NULL)
        return STATUS_ERROR;
    fclose(output_file);

    for(; n > 0; n += step)
    {      
        
        //Create an starting position for N bodies
        FILE* position_file = fopen("../Starting_Configurations/bin_files/random.bin", "wb");
        if(position_file == NULL)
        {
            printf("Error opening output file\n");
            return STATUS_ERROR;
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
        
        //Run simulation for 100 seconds
        double t = run_simulation(100.0, "../Starting_Configurations/bin_files/random.bin");
        
        FILE* output_file = fopen("times.log", "a");
        if(output_file == NULL)
            return STATUS_ERROR;
            
        //Store time
        fprintf(output_file, "%ld %lf\n", n, t);

        //Free memory
        fclose(output_file);

        if(t > max_time)
            break;
    }
    
    return STATUS_OK;
}

double random_num(double min, double max) 
{   
    return min + (max + 1.0)*rand()/(RAND_MAX+1.0);
}