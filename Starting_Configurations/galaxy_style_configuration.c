#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../Metodo_Directo/simulation.h"

#define M_PI acos(-1.0)

int main(int argc, char **argv )
/**
 * This file is to generate a bin file of N bodies set up as a galaxy. Based on the work of logacube on github
 * 
 * @link https://github.com/logacube/octree-based-N-body-sim
*/
{   
    //Check arguments
    if(argc < 2)
    {
        printf("Usage: ./%s [N]\n", argv[0]);
        return -1;
    }
    
    int n = atoi(argv[1]);
    if(n <= 0)
    {
        printf("N must be a positive integer\n");
        return -1;
    }

    //Create an starting position for N bodies
    FILE* position_file = fopen("../Starting_Configurations/bin_files/galaxy.bin", "wb");
    if(position_file == NULL)
    {
        printf("Error opening output file\n");
        return -1;
    }

    double values[8];
    for (int i = 0; i < n; i++) 
    {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX); 
        double radius = (((1.0 / 600.0) * n) + (10.0 / 3.0)) * (0.2 + ((rand() / (double)RAND_MAX))); 


        // Position
        values[0] = radius * cos(angle); // x
        values[1] = radius * sin(angle); // y
        values[2] = ((rand() / (double)RAND_MAX) - 0.5) * 0.1; // z 

        // Mass
        values[3] = 1.0;

        double vel = sqrt((G * n) / radius) * 0.035;
        //double vel = sqrt(G  * radius);

        values[4] = -values[1]  * vel;  //vx
        values[5] = values[0]  * vel;  //vy
        values[6] = 0.0;  //vz

        values[7] = 1/log2(n);              //radius

        fwrite(values, sizeof(values), 1, position_file);
    }

    fclose(position_file);    
}