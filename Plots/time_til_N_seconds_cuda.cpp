#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

#include "../Metodo_Directo/simulation.h"

double random_num(double min, double max);

int main(int argc, char **argv)
/**
 * Small code to generete a log file with the times
 * it takes to simulation to execute a 100 second simulation 
 * with diferent number of bodies
 * @param Starting_N (long): The starting size of N to test
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
    if(argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <Starting_N> <Max_N> <Steps>\n";
        return STATUS_ERROR;
    }
    long n = std::atol(argv[1]);
    double max_time = std::strtod(argv[2], NULL);
    int step = std::atoi(argv[3]);

    std::ofstream output_file;
    output_file.open("times.log",  ios::out | ios::trunc );
    if(!output_file)
    {
        std::cerr << "Error opening output file\n";
        return STATUS_ERROR;
    }
    output_file.close();

    char pos_f_path[] = "../Starting_Configurations/bin_files/random.bin";
    for(; n > 0; n += step)
    {   
        
        //Create an starting position for N bodies
        FILE* position_file = fopen(pos_f_path, "wb");
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
        
        // Run simulation for 100 seconds
        double t = run_simulation(100.0, pos_f_path);
        
        output_file.open("times.log",  ios::out | ios::app );
        if(!output_file)
        {
            std::cerr << "Error opening output file\n";
            return STATUS_ERROR;
        }
        // Store time
        output_file << n << " " << t << "\n";
        output_file.close();
        
        if(t > max_time)
            break;
    }
    return STATUS_OK;
}

double random_num(double min, double max) 
{   
    return min + (max + 1.0) * std::rand() / (RAND_MAX + 1.0);
}
