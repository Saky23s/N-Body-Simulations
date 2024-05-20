#include <stdio.h>
#include <stdlib.h>

double random_num(double min, double max);

int main(int argc, char **argv)
/**
 * Small code to generete a bin file with a random position of N bodies
 * @param N (int): The number of bodies
*/
{   
    if(argc != 2)
    {
        printf("Usage: %s <N>\n", argv[0]);
        return 0;
    }

    int n = atoi(argv[1]);

    //Create an starting position for N bodies
    FILE* position_file = fopen("../../Starting_Configurations/bin_files/random.bin", "wb");
    if(position_file == NULL)
    {
        printf("Error opening output file\n");
        return 0;
    }

    double values[8];
    for(int i = 0; i < n; i++)
    {
        values[0] = random_num(-100, 100);  //x
        values[1] = random_num(-100, 100);  //y
        values[2] = random_num(-100, 100);  //z
        values[3] = 1;                    //mass
        values[4] = random_num(-100, 100);  //vx
        values[5] = random_num(-100, 100);  //vy
        values[6] = random_num(-100, 100);  //vz
        values[7] = values[3];              //radius

        fwrite(values, sizeof(values), 1, position_file);
    }
    fclose(position_file);
    return 1;
}


double random_num(double min, double max) 
{   
    return min + (max + 1.0)*rand()/(RAND_MAX+1.0);
}