#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ERROR  10e-17 
int main(int argc, char **argv)
{
    if(argc != 4)
    {
        printf("Usage: %s <N> <file1> <file2>\n", argv[0]);
        return 0;
    }

    int n = atoi(argv[1]);
    
    //Create an starting position for N bodies
    FILE* base_file = fopen(argv[2], "rb");
    if(base_file == NULL)
    {
        printf("Error opening %s\n", argv[2]);
        return 0;
    }

    //Create an starting position for N bodies
    FILE* file_to_check = fopen(argv[3], "rb");
    if(file_to_check == NULL)
    {
        printf("Error opening %s\n", argv[3]);
        return 0;
    }

    double base_buffer[3];
    double other_buffer[3];
    double error = 0.0;
    for(int i = 0; i < n; i++)
    {
        //Read one of based file line
        fread(base_buffer, 3, sizeof(double), base_file);
        fread(other_buffer, 3, sizeof(double), file_to_check);
        error += fabs(base_buffer[0] - other_buffer[0]) + fabs(base_buffer[1] - other_buffer[1]) + fabs(base_buffer[2] - other_buffer[2]);
        
    }
    fclose(base_file);
    fclose(file_to_check);
    
    if(error < MAX_ERROR)
        return 1;

    printf("%.15lf\n", error);
    return 0;
}