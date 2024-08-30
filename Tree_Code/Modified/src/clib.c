/** 
 * @file clib.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * Assorted C routines with prototypes in stdinc.h.
 * 
 * Modifies the original work of Joshua E. Barnes to remove features
 * that are not required for this investigation 
 * 
 * Most routines where removed and only cpu time and error were left
 *  
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#include "../inc/stdinc.h"
#include <sys/types.h>
#include <sys/times.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <stdarg.h>


double cputime(void)
/**
 * compute total process CPU time in minutes.
 * 
 * @author 1999 by Joshua E. Barnes, Tokyo, JAPAN.
 */
{
    struct tms buffer;

    if (times(&buffer) == -1)
        error("cputime: times() call failed\n");
    return ((buffer.tms_utime + buffer.tms_stime) / (60.0 * HZ));
}

void error(char* fmt, ...)
/**
 * Scream an error in the error terminal but dont die
 * 
 * @param fmt (char*): The text to scream
 * 
 * @author 1999 by Joshua E. Barnes, Tokyo, JAPAN.
 * @author (slight modifications) Santiago Salas
 */
{
    va_list ap;

    va_start(ap, fmt);
    //Invoke error interface
    vfprintf(stderr, fmt, ap);

    //Drain std error buffer
    fflush(stderr);

    va_end(ap);
}



