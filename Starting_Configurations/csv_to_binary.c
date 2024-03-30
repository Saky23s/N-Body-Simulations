#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv )
/**
 * This file is a small script to turn csv files into binary.
*/
{   
    //Check arguments
    if(argc < 3)
    {
        printf("Usage: ./csv_to_binary [input] [output]\n");
        return -1;
    }
    //Read
    FILE* input_file = fopen(argv[1], "r");
    if(input_file == NULL)
    {   
        printf("Error opening input file %s\n", argv[1]);
        return -1;
    }
    //Write
    FILE* output_file = fopen(argv[2], "wb");
    if(output_file == NULL)
    {
        printf("Error opening output file %s\n", argv[2]);
        return -1;
    }

    int header = 0;
    //Array to store values
    double values[8];

    //Read the whole file
    while(!feof(input_file))
    {   
        //skip header
        if(header == 0)
        {   
            //skip header line
            fscanf(input_file, "%*[^\n]\n");
            header = 1;
            continue;
        }
        
        //Read values from csv
        fscanf(input_file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7]);

        //Write values to bin file
        fwrite(values,8,sizeof(double), output_file);
    }
    //close files
    fclose(input_file);
    fclose(output_file);
}