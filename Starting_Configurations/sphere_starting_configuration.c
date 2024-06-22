#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M_PI acos(-1.0)

int main(int argc, char **argv )
/**
 * This file is to generate a bin file of N bodies set up as a sphere
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
    FILE* position_file = fopen("../Starting_Configurations/bin_files/sphere.bin", "wb");
    if(position_file == NULL)
    {
        printf("Error opening output file\n");
        return -1;
    }

    if (n == 1)
    {
        double values[8];
        
        values[0] = 0.0;  //x
        values[1] = 0.0;  //y
        values[2] = 0.0;  //z
        values[3] = 1.0;  //mass
        values[4] = 0.0;  //vx
        values[5] = 0.0;  //vy
        values[6] = 0.0;  //vz
        values[7] = values[3];              //radius

        fwrite(values, sizeof(values), 1, position_file);
       
        fclose(position_file);
        return 1;
    }

    double values[8];

    double phi = M_PI * (sqrt(5) - 1.);
    double scalar = sqrt((n) / (M_PI * 2));
    for( int i = 0; i <  n; i ++)
    {
        double y = 1 - (i / (double) (n - 1)) * 2 ;
        double radius = sqrt(1 - y * y);

        double theta = phi * i;

        double x = cos(theta) * radius;
        double z = sin(theta) * radius;

        printf("%d: %lf,%lf,%lf\n",i,x*scalar,y*scalar,z*scalar);
        
        
        values[0] = x*scalar;  //x
        values[1] = y*scalar;  //y
        values[2] = z*scalar;  //z
        values[3] = 1.0;  //mass
        values[4] = 0.0;  //vx
        values[5] = 0.0;  //vy
        values[6] = 0.0;  //vz
        values[7] = 1.0/log2(n);              //radius

        fwrite(values, sizeof(values), 1, position_file);
    }
        
    fclose(position_file);    
}