/** 
 * @file treecode.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * I/O routines for hierarchical N-body code.
 * 
 * Modifies the original work of Joshua E. Barnes to remove features
 * that are not required for this investigation and make the I-O 
 * system work with our existing framework   
 * 
 * This document changed the way that I-O works to that it works with out grafic engine
 * and with our starting configurations
 * 
 * The input has to be N lines with each line containing
 * x,y,z,mass,vx,vy,vz,radius
 * this can be done with csv files or in binary using doubles
 * 
 * Diagnostics are optional using a macro that can be defined in the makefile
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../inc/treecode.h"
#include "../../../Aux/aux.c"
#include <sys/types.h>
#include <sys/stat.h>
#include <strings.h>

//Interal helpers
local int save_values_bin(int file_number);
local int save_values_csv(int file_number);

#ifdef DIAGNOSTICS
local void diagnostics(void);
local int print_diagnostics(void);
#endif

//Diagnositc output variables.
local real mtot;                                /* total mass of system     */
local real etot[3];                             /* Etot, KE, PE of system   */
local matrix keten;                             /* kinetic energy tensor    */
local matrix peten;                             /* potential energy tensor  */
local vector cmpos;                             /* center of mass position  */
local vector cmvel;                             /* center of mass velocity  */
local vector amvec;                             /* angular momentum vector  */
local int filenumber = 0;

int output(void)
/**
 * 
 * This funtion will control the outputs.
 * 
 * In case of DIAGNOSTICS being required it will call the print_diagnostics funtion that will calculate
 * and print the diagnostics
 * 
 * It will also check if its time to save the positions of the bodies to a file, in which case it will and
 * it will schedule the next output 
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    real teff;

    #ifdef DIAGNOSTICS
        if(print_diagnostics() == STATUS_ERROR)
            return STATUS_ERROR;
    #endif

    //Anticipate slightly...
    teff = tnow + dt/8;

    if (teff >= tout)
    {
        if(save_values_bin(filenumber++) == STATUS_ERROR)
            return STATUS_ERROR;

        //Schedule next output
        tout += speed;
    }
    return STATUS_OK;
}

int load_bodies(char* filename)
/**
 * This funtion creates uses the starting values from a file to load the N bodies
 * @param filepath (char*):  a path to the file with the starting data, must be csv or bin file
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    bodyptr p;
    FILE* f = NULL;

    //Error checking
    if(filename == NULL)
        return STATUS_ERROR;
    

    int extention_type = get_extention_type(filename);
    if(extention_type == EXT_CSV)
    {   
        //Open file
        f = fopen(filename, "r");
        if(f == NULL)
            return STATUS_ERROR;
        
        //Get the number of bodies by the number of lines minus the header
        nbody = count_lines_csv(f) - 1;
        if(nbody <= 0)
            return STATUS_ERROR;
        
        //Memory allocation for the arrays
        bodytab = (bodyptr) calloc(nbody * sizeof(body), 1);
        if(bodytab == NULL)
            return STATUS_ERROR;
        //go back to the begining of file
        rewind(f);

        p = bodytab; 

        //For the number of bodies + header
        for(int i = 0; i < nbody + 1; i++)
        {     
            //read header
            if(i == 0)
            {   
                //skip header line
                fscanf(f, "%*[^\n]\n");
                continue;
            }

            //Read body
            if(fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%*f\n", &(Pos(p)[0]), &(Pos(p)[1]), &(Pos(p)[2]), &Mass(p), &(Vel(p)[0]), &(Vel(p)[1]), &(Vel(p)[2])) == EOF)
            {
                printf("Error reading %s\n", filename);
                return STATUS_ERROR;
            }

            //Set type to body
            Type(p) = BODY;
            p++;
        }
        fclose(f);   
    }                       
    else if (extention_type == EXT_BIN)
    {
        //Read as binary
        FILE* f = fopen(filename, "rb");
        if(f == NULL)
            return STATUS_ERROR;
        

        //Get file size
        fseek(f, 0, SEEK_END); 
        long size = ftell(f); 
        fseek(f, 0, SEEK_SET);

        //The number of bodies is the size of the file / size of each body
        nbody = size / (sizeof(double) * 8); 

        //Memory allocation for the arrays
        bodytab = (bodyptr) calloc(nbody * sizeof(body), 1);
        if(bodytab == NULL)
            return STATUS_ERROR;
        
        //Buffer for one body
        double buffer[8];
        
        //Read the whole file
        for (p = bodytab; p < bodytab+nbody; p++)
        {   
            if(fread(buffer,sizeof(buffer),1,f) == 0)
                return STATUS_ERROR;
                
            Pos(p)[0] = buffer[0];  //x
            Pos(p)[1] = buffer[1];  //y
            Pos(p)[2] = buffer[2];  //z
            Mass(p) = buffer[3];    //mass
            Vel(p)[0] = buffer[4];  //vx
            Vel(p)[1] = buffer[5];  //vy
            Vel(p)[2] = buffer[6];  //vz

            //Buffer[7] is radius, currently useless for data, only useful for graphics

            //Set type to body
            Type(p) = BODY;
        }
        fclose(f);
    }
    //File type not recognized
    else
    {
        return STATUS_ERROR;
    }

    //Set time to 0
    tnow = 0.0;

    //Start root w/ unit cube
    rsize = 1.0;
    
    //Begin counting steps
    nstep = 0;
    //Schedule first output for now
    tout = tnow;

    return STATUS_OK;
}

local int save_values_bin(int file_number)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a bin
 * The file will be stored as /dev/shm/data/FILE_NUMBER.bin
 * @param file_number (int) the file number to be used 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    char filename[256];
    bodyptr p;

    //Construct output name
    snprintf(filename, 256, "/dev/shm/data/%d.bin", file_number);
    
    //Open file
    FILE* f = fopen(filename, "wb");
    if(f == NULL)
        return STATUS_ERROR;
    
    //Print all bodies
    for (p = bodytab; p < bodytab+nbody; p++)
    {
        if (fwrite((void *)  Pos(p), sizeof(real), NDIM, f) != NDIM)
        {
            printf("out_vector: fwrite failed\n");
            return STATUS_ERROR;
        }
    }

    //Close up output file 
    fclose(f);
    return STATUS_OK;
}

local int save_values_csv(int file_number)
/**
 * This funtion will print to the file f the current positions of all the bodies in the simulation as a csv
 * The file will be stored as /dev/shm/data/FILE_NUMBER.csv
 * @param file_number (int) the filenumber to be used 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{   
    char filename[256];
    bodyptr p;

    //Construct output name
    snprintf(filename, 256, "/dev/shm/data/%d.csv", file_number);
    
    //Open file
    FILE* f = fopen(filename, "w");
    if(f == NULL)
        return STATUS_ERROR;

    //For all n bodies
    for (p = bodytab; p < bodytab+nbody; p++)
    {      
        //Print body as csv x,y,z
        if(fprintf(f, "%lf,%lf,%lf\n", Pos(p)[0], Pos(p)[1], Pos(p)[2]) < 0)
            return STATUS_ERROR;
    }

    fclose(f);
    return STATUS_OK;
}

#ifdef DIAGNOSTICS

local void diagnostics(void)
/**
 * Compute set of dynamical diagnostics.
 * @author 2001 by Joshua E. Barnes, Honolulu, Hawai`i.
 */
{
    register bodyptr p;
    real velsq;
    vector tmpv;
    matrix tmpt;

    //Zero total mass
    mtot = 0.0;
    //Zero total KE and PE
    etot[1] = etot[2] = 0.0;
    //Zero ke tensor
    CLRM(keten);
    //Zero pe tensor
    CLRM(peten);
    //Zero am vector 
    CLRV(amvec);
    //Zero c. of m. position   
    CLRV(cmpos);
    //Zero c. of m. velocity
    CLRV(cmvel);

    //Loop over all particles
    for (p = bodytab; p < bodytab+nbody; p++)
    { 
        //Sum particle masses
        mtot += Mass(p);
        //Square vel vector        
        DOTVP(velsq, Vel(p), Vel(p));
        //Sum current KE 
        etot[1] += 0.5 * Mass(p) * velsq;
        //Sum current PE 
        etot[2] += 0.5 * Mass(p) * Phi(p);
        //Sum 0.5 m v_i v_j  
        MULVS(tmpv, Vel(p), 0.5 * Mass(p));
        //
        OUTVP(tmpt, tmpv, Vel(p));
        ADDM(keten, keten, tmpt);
        //Sum m r_i a_j
        MULVS(tmpv, Pos(p), Mass(p));
        OUTVP(tmpt, tmpv, Acc(p));
        ADDM(peten, peten, tmpt);
        //Sum angular momentum
        CROSSVP(tmpv, Vel(p), Pos(p));
        MULVS(tmpv, tmpv, Mass(p));
        ADDV(amvec, amvec, tmpv);
        //Sum cm position  
        MULVS(tmpv, Pos(p), Mass(p));
        ADDV(cmpos, cmpos, tmpv);
        //Sum cm momentum
        MULVS(tmpv, Vel(p), Mass(p));
        ADDV(cmvel, cmvel, tmpv);
    }
    //Sum KE and PE 
    etot[0] = etot[1] + etot[2];
    //Normalize cm coords
    DIVVS(cmpos, cmpos, mtot);
    DIVVS(cmvel, cmvel, mtot);
}

local int print_diagnostics(void)
/**
 * Funtion that call antoher funtion to calculate diagnostics and then prints
 * the important values to terminal
 */
{   
    real cmabs, amabs;

    //Compute std diagnostics
    diagnostics();
    //Find magnitude of cm vel
    ABSV(cmabs, cmvel);
    //Find magnitude of J vect
    ABSV(amabs, amvec);

    //Print values
    printf("\n    %8s%8s%8s%8s%8s%8s%8s%8s\n",
           "time", "|T+U|", "T", "-U", "-T/U", "|Vcom|", "|Jtot|", "CPUtot");
    printf("    %8.3f%8.5f%8.5f%8.5f%8.5f%8.5f%8.5f%8.3f\n",
           tnow, ABS(etot[0]), etot[1], -etot[2], -etot[1]/etot[2],
           cmabs, amabs, cputime());
    
    return STATUS_OK;
}

void forcereport(void)
/**
 * Funtion that print staristics on tree construction and force calculation.
 * @author 2001 by Joshua E. Barnes, Honolulu, Hawai`i.
 */
{
    printf("\n\t%8s%8s%8s%8s%10s%10s%8s\n",
           "rsize", "tdepth", "ftree",
           "actmax", "nbbtot", "nbctot", "CPUfc");
    printf("\t%8.1f%8d%8.3f%8d%10d%10d%8.3f\n",
           rsize, tdepth, (nbody + ncell - 1) / ((real) ncell),
           actmax, nbbcalc, nbccalc, cpuforce);
}

#endif