#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NO_EXT
  #define NO_EXT -1
  #define EXT_CSV 0
  #define EXT_BIN 1
#endif

int get_extention_type(const char *filename) ;

int main(int argc, char **argv )
/**
 * This file is a small script to create the starting position file for the 
 * graphic engine to work. It takes the starting data file used in the 
 * data generation as an argument and creates a file that the graphic engine will use.
 * 
 * Works with csv files and bin files
*/
{ 
    if(argc < 2)
    {
        printf("Usage: ./graphic_starting_position [input]\n");
        return -1;
    }
    int ext = get_extention_type(argv[1]);

    if(ext == EXT_CSV)
    {
        FILE* input_file = fopen(argv[1], "r");
        if(input_file == NULL)
        {   
            printf("Error opening input file %s\n", argv[1]);
            return -1;
        }

        FILE* output_file = fopen("../Graphics/data/starting_positions.bin", "w");
        if(output_file == NULL)
        {
            printf("Error opening output file ../Graphics/data/starting_positions.bin\n");
            return -1;
        }

        int header = 0;
        //Array to store values
        double values[8];
        double buffer[4];


        //Read the whole file
        while(!feof(input_file))
        {   
            //read header
            if(header == 0)
            {   
                //skip header line
                fscanf(input_file, "%*[^\n]\n");
                header = 1;
                continue;
            }
            
            //Read values from csv
            fscanf(input_file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7]);

            buffer[0] = values[0]; //x
            buffer[1] = values[1]; //y
            buffer[2] = values[2]; //z
            buffer[3] = values[7]; //radius

            //Write values to bin file
            fwrite(buffer, sizeof(buffer), 1, output_file);
        }
        //close files
        fclose(input_file);
        fclose(output_file);
        return 1;
    }
    else if(ext == EXT_BIN)
    {
        FILE* input_file = fopen(argv[1], "rb");
        if(input_file == NULL)
        {   
            printf("Error opening input file %s\n", argv[1]);
            return -1;
        }

        FILE* output_file = fopen("../Graphics/data/starting_positions.bin", "wb");
        if(output_file == NULL)
        {
            printf("Error opening output file ../Graphics/data/starting_positions.bin\n");
            return -1;
        }

        //Array to store values
        double values[8];
        double buffer[4];
        //Read the whole file
        while(!feof(input_file))
        {   
            
            //Read values from bin
            if(fread(values, sizeof(values), 1, input_file) == 0)
                break;
            
            buffer[0] = values[0]; //x
            buffer[1] = values[1]; //y
            buffer[2] = values[2]; //z
            buffer[3] = values[7]; //radius

            //Write values to bin file
            fwrite(buffer, sizeof(buffer), 1, output_file);
        }
        //close files
        fclose(input_file);
        fclose(output_file);
        return 1;
    }
    else
    {
        printf("Invalid file extention\n");
        return -1;
    }
   
}

int get_extention_type(const char *filename) 
/**
 * This funtion will check if a filename is csv, bin or other extention type
 * @param filename (char *):  the filename we are checking
 * @param file_type(int): 
 *                          -1 (NO_EXT) extention not valid
 *                           0 (EXT_CSV) csv file
 *                           1 (EXT_BIN) bin file
 */
{
    const char *dot = strrchr(filename, '.');
    //No extention
    if(!dot || dot == filename) return NO_EXT;

    //If extention is csv
    if(strcmp(dot + 1, "csv") == 0) return EXT_CSV;

    //If extention is bin
    if(strcmp(dot + 1, "bin") == 0) return EXT_BIN;

    //Extention not recognised
    return NO_EXT;
}