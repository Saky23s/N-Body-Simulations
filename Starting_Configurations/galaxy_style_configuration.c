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
        double curve = 2 * M_PI * (rand() / (double) RAND_MAX);
        double radius = 30 * (rand() / (double) RAND_MAX);
        double x = (cos(curve) - sin(curve));
        double y = (cos(curve) + sin(curve));


        values[0] = radius * x; //x
        values[1] = radius * y; //y
        values[2] = 0.0; // z 

        // Mass
        if ( i % 10 == 0 && i % 100 != 0)
            values[3] = 2;
        else if (i % 100 == 0)
            values[3] = 3;
        else
            values[3] = 1;

        double vel = sqrt((G * n) / radius) * 0.0035;
        //double vel = sqrt(G  * radius);

        values[4] = -values[1]  * vel;  //vx
        values[5] = values[0]  * vel;  //vy
        values[6] = 0.0;  //vz

        if ( i % 10 == 0 && i % 100 != 0)
            values[7] = 2 /log2(n);              //radius
        else if (i % 100 == 0)
            values[7] = 3 /log2(n);              //radius
        else
            values[7] = 1 /log2(n);              //radius

        fwrite(values, sizeof(values), 1, position_file);
    }

    fclose(position_file);    
}