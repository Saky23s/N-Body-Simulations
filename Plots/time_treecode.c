#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../Tree_Code/Modified/inc/stdinc.h"

#define PI         3.14159265358979323846

/*
 * TESTDATA: generate Plummer model initial conditions for test runs,
 * scaled to units such that M = -4E = G = 1 (Henon, Hegge, etc).
 * See Aarseth, SJ, Henon, M, & Wielen, R (1974) Astr & Ap, 37, 183.
 */

#define MFRAC  0.999                            /* cut off 1-MFRAC of mass  */
void pickshell(double vec[], double rad);
double xrandom(double xl, double xh);
int create_position_file(int n);

int main(int argc, char **argv)
/**
 * Small code to generete a log file with the times
 * it takes to simulation to execute a 100 second simulation 
 * with diferent number of bodies
 * @param Starting_N (long): The starting size of N to test
 * @param Max_N (long): The maximum size of N to test
 * @param steps (int): The increment between timing
 * @return 'times.log' will have all the results stored at the end
 * of the excution, in format:
 * 
 * n t
 * 
 * where N is the number of bodies that simulation had
 * and t is the time it took to excute
*/
{
    if(argc != 4)
    {
        printf("Usage: %s <Starting_N> <Max_N> <Steps>\n", argv[0]);
        return STATUS_ERROR;
    }
    long n = atol(argv[1]);
    long max_n = atol(argv[2]);
    int step = atoi(argv[3]);

    //Create an empty file
    FILE* output_file = fopen("times.log", "w");
    if(output_file == NULL)
        return STATUS_ERROR;
    fclose(output_file);

    for(; n <= max_n; n += step)
    {      
        create_position_file(n);
        
        //Run simulation for 100 seconds
        double t = run_simulation(100.0, "../Starting_Configurations/bin_files/plummer.bin");
        
        FILE* output_file = fopen("times.log", "a");
        if(output_file == NULL)
            return STATUS_ERROR;
            
        //Store time
        fprintf(output_file, "%ld %lf\n", n, t);

        //Free memory
        fclose(output_file);
    }
    
    return STATUS_OK;
}

int create_position_file(int n)
{
    srand(time(NULL));

    //Create an starting position for N bodies
    FILE* position_file = fopen("../Starting_Configurations/bin_files/plummer.bin", "wb");
    if(position_file == NULL)
    {
        printf("Error opening output file\n");
        return STATUS_ERROR;
    }
    
    double rsc, vsc, r, v, x, y;
    double rcm[3] = {0.0,0.0,0.0};
    double vcm[3] = {0.0,0.0,0.0};
    double values[8];

    /* length scale factor  */
    rsc = (3 * PI) / 16;

    /* find speed scale factor  */                      
    vsc = sqrt(1.0 / rsc);   

    for (int i = 0; i < n; i++) 
    {
        x = xrandom(0.0, MFRAC);                /* pick enclosed mass       */
        r = 1.0 / sqrt(pow(x, -2.0/3.0) - 1); /* find enclosing radius    */
        pickshell(&(values[0]), rsc * r);       /* pick position vector     */

        do 
        {                                       /* select from fn g(x)      */
            x = xrandom(0.0, 1.0);              /* for x in range 0:1       */
            y = xrandom(0.0, 0.1);              /* max of g(x) is 0.092     */
        } while (y > x*x * pow(1 - x*x, 3.5)); /* using von Neumann tech   */

        v = x * sqrt(2.0 / sqrt(1 + r*r));    /* find resulting speed     */

        pickshell(&(values[4]), vsc * v);       /* pick velocity vector     */

        /* accumulate cm position   */
        rcm[0] += values[0] * (1.0 / n);                                             
        rcm[1] += values[1] * (1.0 / n);                                             
        rcm[2] += values[2] * (1.0 / n);                                             
        /* accumulate cm velocity   */
        
        vcm[0] += values[4] * (1.0 / n);                                             
        vcm[1] += values[5] * (1.0 / n);                                             
        vcm[2] += values[6] * (1.0 / n);
    }

    for (int i = 0; i < n; i++) 
    {
        x = xrandom(0.0, MFRAC);                /* pick enclosed mass       */
        r = 1.0 / sqrt(pow(x, -2.0/3.0) - 1); /* find enclosing radius    */
        pickshell(&(values[0]), rsc * r);       /* pick position vector     */

        do 
        {                                       /* select from fn g(x)      */
            x = xrandom(0.0, 1.0);              /* for x in range 0:1       */
            y = xrandom(0.0, 0.1);              /* max of g(x) is 0.092     */
        } while (y > x*x * pow(1 - x*x, 3.5)); /* using von Neumann tech   */

        v = x * sqrt(2.0 / sqrt(1 + r*r));    /* find resulting speed     */

        pickshell(&(values[4]), vsc * v);       /* pick velocity vector     */

        values[0] = values[0] - rcm[0];
        values[1] = values[1] - rcm[1];
        values[2] = values[2] - rcm[2];
        values[3] = 1.0 / n;
        values[4] = values[4] - vcm[0];
        values[5] = values[5] - vcm[1];
        values[6] = values[6] - vcm[2];
        values[7] = 1.0 / log2(n);
        
        fwrite(values, sizeof(values), 1, position_file);
    }
    fclose(position_file);
}

/*
 * PICKSHELL: pick point on shell.
 */
void pickshell(double vec[], double rad)
{
    double rsq, rscale;
    int i;
    int ndim = 3;

    do 
    {
        rsq = 0.0;
        for (i = 0; i < ndim; i++) 
        {
            vec[i] = xrandom(-1.0, 1.0);
            rsq = rsq + vec[i] * vec[i];
        }
    } while (rsq > 1.0);

    rscale = rad / sqrt(rsq);

    for (i = 0; i < ndim; i++)
    {
        vec[i] = vec[i] * rscale;
    }
}

/*
 * XRANDOM: floating-point random number routine.
 */

double xrandom(double xl, double xh)
{

    return (xl + (xh - xl) * ((double) rand()) / 2147483647.0);
}