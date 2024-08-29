/****************************************************************************/
/* TREEIO.C: I/O routines for hierarchical N-body code. Public routines:    */
/* inputdata(), startoutput(), output(), savestate(), restorestate().       */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../inc/getparam.h"
#include "../inc/treecode.h"
#include "../../../Aux/aux.c"
#include <sys/types.h>
#include <sys/stat.h>
#include <strings.h>

/*
 * Prototypes for local routines.
 */

local void outputdata(void);                    /* write N-body data        */
local void diagnostics(void);                   /* eval N-body diagnostics  */
local void in_int(stream, int *);               /* input integer value      */
local void in_real(stream, real *);             /* input real value         */
local void in_vector(stream, vector);           /* input vector of reals    */
local void out_int(stream, int);                /* output integer value     */
local void out_real(stream, real);              /* output real value        */
local void out_vector(stream, vector);          /* output vector of reals   */
int save_values_bin(int file_number);
int load_bodies(char* filename);

/*
 * Diagnositc output variables.
 */

local real mtot;                                /* total mass of system     */
local real etot[3];                             /* Etot, KE, PE of system   */
local matrix keten;                             /* kinetic energy tensor    */
local matrix peten;                             /* potential energy tensor  */
local vector cmpos;                             /* center of mass position  */
local vector cmvel;                             /* center of mass velocity  */
local vector amvec;                             /* angular momentum vector  */
local int filenumber = 0;

/*
 * INPUTDATA: read initial conditions from input file.
 */

void inputdata(void)
{
    load_bodies(infile);
}

/*
 * STARTOUTPUT: begin output to log file.
 */

void startoutput(void)
{
    printf("\n%s\n", headline);                 /* print headline, params   */
#if defined(USEFREQ)
    printf("\n%8s%10s%10s", "nbody", "freq", "eps");
#else
    printf("\n%8s%10s%10s", "nbody", "dtime", "eps");
#endif
#if !defined(QUICKSCAN)
    printf("%10s", "theta");
#endif
#if defined(USEFREQ)
    printf("%10s%10s%10s\n", "usequad", "freqout", "tstop");
    printf("%8d%10.2f%10.4f", nbody, freq, eps);
#else
    printf("%10s%10s%10s\n", "usequad", "dtout", "tstop");
    printf("%8d%10.5f%10.4f", nbody, dtime, eps);
#endif
#if !defined(QUICKSCAN)
    printf("%10.2f", theta);
#endif
#if defined(USEFREQ)
    printf("%10s%10.2f%10.4f\n", usequad ? "true" : "false", freqout, tstop);
#else
    printf("%10s%10.5f%10.4f\n", usequad ? "true" : "false", dtout, tstop);
#endif
    if (! strnull(options))                     /* print options, if any    */
        printf("\n\toptions: %s\n", options);
    if (! strnull(savefile))                    /* was state file given?    */
        savestate(savefile);                    /* save initial data        */
}

/*
 * FORCEREPORT: print staristics on tree construction and force calculation.
 */

void forcereport(void)
{
    printf("\n\t%8s%8s%8s%8s%10s%10s%8s\n",
           "rsize", "tdepth", "ftree",
           "actmax", "nbbtot", "nbctot", "CPUfc");
    printf("\t%8.1f%8d%8.3f%8d%10d%10d%8.3f\n",
           rsize, tdepth, (nbody + ncell - 1) / ((real) ncell),
           actmax, nbbcalc, nbccalc, cpuforce);
}

/*
 * OUTPUT: compute diagnostics and output body data.
 */

void output(void)
{
    real cmabs, amabs, teff;

    diagnostics();                              /* compute std diagnostics  */
    ABSV(cmabs, cmvel);                         /* find magnitude of cm vel */
    ABSV(amabs, amvec);                         /* find magnitude of J vect */
    printf("\n    %8s%8s%8s%8s%8s%8s%8s%8s\n",
           "time", "|T+U|", "T", "-U", "-T/U", "|Vcom|", "|Jtot|", "CPUtot");
    printf("    %8.3f%8.5f%8.5f%8.5f%8.5f%8.5f%8.5f%8.3f\n",
           tnow, ABS(etot[0]), etot[1], -etot[2], -etot[1]/etot[2],
           cmabs, amabs, cputime());
#if defined(USEFREQ)
    teff = tnow + (freq > 0 ? 0.125/freq : 0);  /* anticipate slightly...   */
#else
    teff = tnow + dtime/8;                      /* anticipate slightly...   */
#endif
    if (teff >= tout)     /* time for data output?    */
        outputdata();
    if (! strnull(savefile))                    /* was state file given?    */
        savestate(savefile);                    /* save data for restart    */
}

/*
 * OUTPUTDATA: output body data.
 */

void outputdata(void)
{
    save_values_bin(filenumber);
    filenumber++;
    printf("\n\tdata output to file %d at time %f\n", filenumber, tnow);
    #if defined(USEFREQ)
    tout += 1.0 / freqout;                      /* schedule next output     */
    #else
        tout += dtout;                              /* schedule next output     */
    #endif
}

/*
 * DIAGNOSTICS: compute set of dynamical diagnostics.
 */

local void diagnostics(void)
{
    register bodyptr p;
    real velsq;
    vector tmpv;
    matrix tmpt;

    mtot = 0.0;                                 /* zero total mass          */
    etot[1] = etot[2] = 0.0;                    /* zero total KE and PE     */
    CLRM(keten);                                /* zero ke tensor           */
    CLRM(peten);                                /* zero pe tensor           */
    CLRV(amvec);                                /* zero am vector           */
    CLRV(cmpos);                                /* zero c. of m. position   */
    CLRV(cmvel);                                /* zero c. of m. velocity   */
    for (p = bodytab; p < bodytab+nbody; p++) { /* loop over all particles  */
        mtot += Mass(p);                        /* sum particle masses      */
        DOTVP(velsq, Vel(p), Vel(p));           /* square vel vector        */
        etot[1] += 0.5 * Mass(p) * velsq;       /* sum current KE           */
        etot[2] += 0.5 * Mass(p) * Phi(p);      /* and current PE           */
        MULVS(tmpv, Vel(p), 0.5 * Mass(p));     /* sum 0.5 m v_i v_j        */
        OUTVP(tmpt, tmpv, Vel(p));
        ADDM(keten, keten, tmpt);
        MULVS(tmpv, Pos(p), Mass(p));           /* sum m r_i a_j            */
        OUTVP(tmpt, tmpv, Acc(p));
        ADDM(peten, peten, tmpt);
        CROSSVP(tmpv, Vel(p), Pos(p));          /* sum angular momentum     */
        MULVS(tmpv, tmpv, Mass(p));
        ADDV(amvec, amvec, tmpv);
        MULVS(tmpv, Pos(p), Mass(p));           /* sum cm position          */
        ADDV(cmpos, cmpos, tmpv);
        MULVS(tmpv, Vel(p), Mass(p));           /* sum cm momentum          */
        ADDV(cmvel, cmvel, tmpv);
    }
    etot[0] = etot[1] + etot[2];                /* sum KE and PE            */
    DIVVS(cmpos, cmpos, mtot);                  /* normalize cm coords      */
    DIVVS(cmvel, cmvel, mtot);
}

/*
 * IN_INT, IN_REAL, IN_VECTOR: low level input routines.
 */

local void in_int(stream str, int *iptr)
{
#if !defined(BINARYIO)
    if (fscanf(str, "%d", iptr) != 1)
        error("in_int: input conversion error\n");
#else
    if (fread((void *) iptr, sizeof(int), 1, str) != 1)
        error("in_int: fread failed\n");
#endif
}

local void in_real(stream str, real *rptr)
{
    double tmp;

#if !defined(BINARYIO)
    if (fscanf(str, "%lf", &tmp) != 1)
        error("in_real: input conversion error\n");
    *rptr = tmp;
#else
    if (fread((void *) rptr, sizeof(real), 1, str) != 1)
        error("in_real: fread failed\n");
#endif
}

local void in_vector(stream str, vector vec)
{
    double tmpx, tmpy, tmpz;

#if !defined(BINARYIO)
    if (fscanf(str, "%lf%lf%lf", &tmpx, &tmpy, &tmpz) != 3)
        error("in_vector: input conversion error\n");
    vec[0] = tmpx;
    vec[1] = tmpy;
    vec[2] = tmpz;
#else
    if (fread((void *) vec, sizeof(real), NDIM, str) != NDIM)
        error("in_vector: fread failed\n");
#endif
}

/*
 * OUT_INT, OUT_REAL, OUT_VECTOR: low level output routines.
 */

#define IFMT  " %d"                             /* output format for ints   */
#define RFMT  " %14.7E"                         /* output format for reals  */

local void out_int(stream str, int ival)
{
#if !defined(BINARYIO)
    if (fprintf(str, IFMT "\n", ival) < 0)
        error("out_int: fprintf failed\n");
#else
    if (fwrite((void *) &ival, sizeof(int), 1, str) != 1)
        error("out_int: fwrite failed\n");
#endif
}

local void out_real(stream str, real rval)
{
#if !defined(BINARYIO)
    if (fprintf(str, RFMT "\n", rval) < 0)
        error("out_real: fprintf failed\n");
#else
    if (fwrite((void *) &rval, sizeof(real), 1, str) != 1)
        error("out_real: fwrite failed\n");
#endif
}

local void out_vector(stream str, vector vec)
{
#if !defined(BINARYIO)
    if (fprintf(str, RFMT RFMT RFMT "\n", vec[0], vec[1], vec[2]) < 0)
        error("out_vector: fprintf failed\n");
#else
    if (fwrite((void *) vec, sizeof(real), NDIM, str) != NDIM)
        error("out_vector: fwrite failed\n");
#endif
}

/*
 * SAVESTATE: write current state to disk file.
 */

#define safewrite(ptr,len,str)                  \
    if (fwrite((void *) ptr, len, 1, str) != 1) \
        error("savestate: fwrite failed\n")

void savestate(string pattern)
{
    char namebuf[256];
    stream str;
    int nchars;

    sprintf(namebuf, pattern, nstep & 1);       /* construct alternate name */
    str = stropen(namebuf, "w!");
    nchars = strlen(getargv0()) + 1;
    safewrite(&nchars, sizeof(int), str);
    safewrite(getargv0(), nchars * sizeof(char), str);
    nchars = strlen(getversion()) + 1;
    safewrite(&nchars, sizeof(int), str);
    safewrite(getversion(), nchars * sizeof(char), str);
#if defined(USEFREQ)
    safewrite(&freq, sizeof(real), str);
#else
    safewrite(&dtime, sizeof(real), str);
#endif
#if !defined(QUICKSCAN)
    safewrite(&theta, sizeof(real), str);
#endif
    safewrite(&usequad, sizeof(bool), str);
    safewrite(&eps, sizeof(real), str);
    nchars = strlen(options) + 1;
    safewrite(&nchars, sizeof(int), str);
    safewrite(options, nchars * sizeof(char), str);
    safewrite(&tstop, sizeof(real), str);
#if defined(USEFREQ)
    safewrite(&freqout, sizeof(real), str);
#else
    safewrite(&dtout, sizeof(real), str);
#endif
    safewrite(&tnow, sizeof(real), str);
    safewrite(&tout, sizeof(real), str);
    safewrite(&nstep, sizeof(int), str);
    safewrite(&rsize, sizeof(real), str);
    safewrite(&nbody, sizeof(int), str);
    safewrite(bodytab, nbody * sizeof(body), str);
    fclose(str);
}

/*
 * RESTORESTATE: restore state from disk file.
 */

#define saferead(ptr,len,str)                  \
    if (fread((void *) ptr, len, 1, str) != 1) \
        error("restorestate: fread failed\n")

void restorestate(string file)
{
    stream str;
    int nchars;
    string program, version;

    str = stropen(file, "r");
    saferead(&nchars, sizeof(int), str);
    program = (string) allocate(nchars * sizeof(char));
    saferead(program, nchars * sizeof(char), str);
    saferead(&nchars, sizeof(int), str);
    version = (string) allocate(nchars * sizeof(char));
    saferead(version, nchars * sizeof(char), str);
    if (! streq(program, getargv0()) ||         /* check program, version   */
          ! streq(version, getversion()))
        printf("warning: state file may be outdated\n\n");
#if defined(USEFREQ)
    saferead(&freq, sizeof(real), str);
#else
    saferead(&dtime, sizeof(real), str);
#endif
#if !defined(QUICKSCAN)
    saferead(&theta, sizeof(real), str);
#endif
    saferead(&usequad, sizeof(bool), str);
    saferead(&eps, sizeof(real), str);
    saferead(&nchars, sizeof(int), str);
    options = (string) allocate(nchars * sizeof(char));
    saferead(options, nchars * sizeof(char), str);
    saferead(&tstop, sizeof(real), str);
#if defined(USEFREQ)
    saferead(&freqout, sizeof(real), str);
#else
    saferead(&dtout, sizeof(real), str);
#endif
    saferead(&tnow, sizeof(real), str);
    saferead(&tout, sizeof(real), str);
    saferead(&nstep, sizeof(int), str);
    saferead(&rsize, sizeof(real), str);
    saferead(&nbody, sizeof(int), str);
    bodytab = (bodyptr) allocate(nbody * sizeof(body));
    saferead(bodytab, nbody * sizeof(body), str);
    fclose(str);
}


int save_values_bin(int file_number)
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
        return -1;
    
    //Print all bodies
    for (p = bodytab; p < bodytab+nbody; p++)
    {
        if (fwrite((void *)  Pos(p), sizeof(real), NDIM, f) != NDIM)
        {
            printf("out_vector: fwrite failed\n");
            return -1;
        }
    }

    //Close up output file 
    fclose(f);
    return 1;
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
        return -1;
    

    int extention_type = get_extention_type(filename);
    if(extention_type == EXT_CSV)
    {   
        //Open file
        f = fopen(filename, "r");
        if(f == NULL)
            return -1;
        
        //Get the number of bodies by the number of lines minus the header
        nbody = count_lines_csv(f) - 1;
        if(nbody <= 0)
            return -1;
        
        //Memory allocation for the arrays
        bodytab = (bodyptr) calloc(nbody * sizeof(body), 1);
        if(bodytab == NULL)
            return -1;
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
                return -1;
            }
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
            return -1;
        

        //Get file size
        fseek(f, 0, SEEK_END); 
        long size = ftell(f); 
        fseek(f, 0, SEEK_SET);

        //The number of bodies is the size of the file / size of each body
        nbody = size / (sizeof(double) * 8); 

        //Memory allocation for the arrays
        bodytab = (bodyptr) calloc(nbody * sizeof(body), 1);
        if(bodytab == NULL)
            return -1;
        
        //Buffer for one body
        double buffer[8];
        
        //Read the whole file
        for (p = bodytab; p < bodytab+nbody; p++)
        {   
            if(fread(buffer,sizeof(buffer),1,f) == 0)
                return -1;
                
            Pos(p)[0] = buffer[0];  //x
            Pos(p)[1] = buffer[1];  //y
            Pos(p)[2] = buffer[2];  //z
            Mass(p) = buffer[3];    //mass
            Vel(p)[0] = buffer[4];  //vx
            Vel(p)[1] = buffer[5];  //vy
            Vel(p)[2] = buffer[6];  //vz

            //Buffer[7] is radius, currently useless for data, only useful for graphics
            Type(p) = BODY;
        }
        fclose(f);
    }
    //File type not recognized
    else
    {
        return -1;
    }
    tnow = 0.0;
    return 1;
}