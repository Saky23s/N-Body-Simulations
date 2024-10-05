#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../Metodo_Directo/simulation.h"

#define M_PI acos(-1.0)

int main(int argc, char **argv )
/**
 * This file is to generate a bin file of N bodies set up as a galaxy 
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
        double curve = 2* M_PI * (rand() / (double) RAND_MAX);
        double radius = 30 * (rand() / (double) RAND_MAX);
        double x = (cos(curve) - sin(curve));
        double y = (cos(curve) + sin(curve));


        values[0] = radius * x; //x
        values[1] = radius * y; //y
        values[2] =  0.02 * ((rand() /(double) RAND_MAX) - 0.5); //z

        double vel = sqrt((G * n) / radius) *0.025;

        values[3] = 1.0;  //mass

        values[4] = -y * vel;  //vx
        values[5] = x * vel;  //vy
        values[6] = 0.0;  //vz

        values[7] = 1/log2(n);              //radius

        fwrite(values, sizeof(values), 1, position_file);
    }
    fclose(position_file);    

}