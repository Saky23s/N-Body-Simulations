#include <stdio.h>
#include <stdlib.h>

//Helper macrowords
#ifndef NO_EXT
  #define NO_EXT -1
  #define EXT_CSV 0
  #define EXT_BIN 1
#endif

#ifndef BUF_SIZE
    #define BUF_SIZE 65536
#endif

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

int count_lines_csv(FILE* f)
/**
 * This returns the number of lines in a text file, is O(N) but its the only way
 * @param file (FILE*) an openned file which lines are going to be read
 * @return n_lines (int) the number of lines in the file
 */
{   
    //Buffer to read all data too
    char buffer[BUF_SIZE];
    int n_lines = 0;
    //Til we read the whole file
    for(;;)
    {   
        //Load a lot of characters 
        size_t res = fread(buffer, 1, BUF_SIZE, f);
        if (ferror(f))
            return -1;

        //Iterate the characters and count the \n
        int i;
        for(i = 0; i < res; i++)
            if (buffer[i] == '\n')
                n_lines++;

        //If we reached the end of the file stop reading
        if (feof(f))
            break;
        //If we havent reached the end keep going
    }

    return n_lines;
}