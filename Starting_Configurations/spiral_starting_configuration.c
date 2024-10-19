#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../Metodo_Directo/simulation.h"

#define M_PI acos(-1.0)

int main(int argc, char **argv )
/**
 * This file is to generate a bin file of N bodies set up as a spiral 
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
    FILE* position_file = fopen("../Starting_Configurations/bin_files/spiral.bin", "wb");
    if(position_file == NULL)
    {
        printf("Error opening output file\n");
        return -1;
    }

    double amplitud = 0.6325/sqrt(n);
    
    double values[8];
    for (int i = 0; i < n; i++) 
    {   
        double x = amplitud * i * cos(i);
        double y = amplitud * i * sin(i);

        values[0] = x; //x
        values[1] = y; //y
        values[2] =  0.0; //z


        //Calculate distance to the center
        double distance = (sqrt(x * x + y * y));
        values[3] = 1.0 + distance * 0.0;  //mass
        
        double baseVelocity = 1.0; 
        double velocityScale = 3.0; 

        //Calculate vector to the center
        double dx = 0.0 - x;
        double dy = 0.0 - y;
        
        double velocityFactor = velocityScale * exp(-distance / 7.0); // Adjust the decay rate
        double velMagnitude = baseVelocity * velocityFactor;
        values[4] = dy * velMagnitude;  //vx
        values[5] = -dx * velMagnitude;  //vy
        values[6] = 0.0;  //vz

        values[7] = values[3]/log2(n);              //radius

        fwrite(values, sizeof(values), 1, position_file);
    }
    fclose(position_file);    

}

