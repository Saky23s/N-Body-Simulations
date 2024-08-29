/****************************************************************************/
/* CLIB.C: assorted C routines with prototypes in stdinc.h.                 */
/* Copyright (c) 1999 by Joshua E. Barnes, Tokyo, JAPAN.                    */
/****************************************************************************/

#include "../inc/stdinc.h"
#include <sys/types.h>
#include <sys/times.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <stdarg.h>

/*
 * ALLOCATE: memory allocation, with error checking.
 */

void *allocate(int nb)
{
    void *mem;

    mem = calloc(nb, 1);                /* allocate, also clearing memory   */
    if (mem == NULL)
        error("allocate: not enuf memory (%d bytes)\n", nb);
    return (mem);
}

/*
 * CPUTIME: compute total process CPU time in minutes.
 */

double cputime(void)
{
    struct tms buffer;

    if (times(&buffer) == -1)
        error("cputime: times() call failed\n");
    return ((buffer.tms_utime + buffer.tms_stime) / (60.0 * HZ));
}

/*
 * ERROR: scream and die quickly.
 */

void error(string fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);          /* invoke interface to printf       */
    fflush(stderr);                     /* drain std error buffer           */
    va_end(ap);
    exit(1);                            /* quit with error status           */
}

/*
 * EPRINTF: scream, but don't die yet.
 */

void eprintf(string fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);          /* invoke interface to printf       */
    fflush(stderr);                     /* drain std error buffer           */
    va_end(ap);
}

/*
 * SCANOPT: scan string of the form "word1,word2,..." for a match.
 * Words must be separated by commas only -- no spaces allowed!
 */

bool scanopt(string opt, string key)
{
    char *op, *kp;

    op = (char *) opt;                  /* start scan of option strings     */
    while (*op != NULL) {               /* loop while words left to check   */
        kp = key;                       /* (re)start scan of key word       */
        while ((*op != ',' ? *op : (char) NULL) == *kp) {
                                        /* char by char, compare word, key  */
            if (*kp++ == NULL)          /* reached end of key word, so...   */
                return (TRUE);          /* indicate success                 */
            op++;                       /* else go on to next char          */
        }
        while (*op != NULL && *op++ != ',')
                                        /* scan for start of next word      */
            continue;
    }
    return (FALSE);                     /* indicate failure                 */
}
