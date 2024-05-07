#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include "../Solucion_Ingenua/inc/simulation.h"

double random_num(double min, double max);

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <Max_N> <Steps>\n";
        return STATUS_ERROR;
    }
    long max_n = std::atol(argv[1]);
    int step = std::atoi(argv[2]);

    std::ofstream output_file;
    output_file.open("times.log");
    if(!output_file)
    {
        std::cerr << "Error opening output file\n";
        return STATUS_ERROR;
    }

    for(long n = 1; n <= max_n; n += step)
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
        
        // Create a simulation for that file
        Simulation *simulation = load_bodies("../Starting_Configurations/bin_files/random.bin");
        if(simulation == nullptr) 
        {
            std::cout << "Error while loading simulation\n";
            return STATUS_ERROR;
        }
        
        // Run simulation for 100 seconds
        double t = run_simulation(simulation, 100.0);
        
        // Store time
        output_file << n << " " << t << "\n";

        // Free memory
        free_simulation(simulation);
    }
    output_file.close();
    return STATUS_OK;
}

double random_num(double min, double max) 
{   
    return min + (max + 1.0) * std::rand() / (RAND_MAX + 1.0);
}
