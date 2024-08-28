/****************************************************************************/
/* STDINC.H: standard include file for Zeno C programs.                     */
/* Copyright (c) 1999 by Joshua E. Barnes, Tokyo, JAPAN.                    */
/****************************************************************************/

#ifndef _stdinc_h
#define _stdinc_h

/*
 * Always include stdio.h and stdlib.h.
 */

#include <stdio.h>
#include <stdlib.h>

/*
 * NULL: value for null pointers, normally defined by stdio.h.
 */

#if !defined(NULL)
#define NULL 0L
#endif

/*
 * LOCAL: synonym for static declares an object as local to a source file.
 */

#define local     static

/*
 * BOOL, TRUE, FALSE: standard names for logical values.
 */

typedef short int bool;

#if !defined(TRUE)
#define TRUE  ((bool) 1)
#define FALSE ((bool) 0)
#endif

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

/*
 * BYTE: name a handy chunk of bits.
 */

typedef unsigned char byte;

/*
 * STRING: null-terminated char array.
 */

typedef char *string;

/*
 * STREAM: more elegant synonym for FILE *.
 */

typedef FILE *stream;                   /* note: stdio.h is included above  */


/*
 * Default precision:  DOUBLEPREC 
 */
#define DOUBLEPREC
#if defined(DOUBLEPREC)
typedef double real, *realptr;
#define Precision "DOUBLEPREC"
#endif

/*
 * PI, etc.  --  mathematical constants.
 */

#define PI         3.14159265358979323846
#define TWO_PI     6.28318530717958647693
#define FOUR_PI   12.56637061435917295385
#define HALF_PI    1.57079632679489661923
#define FRTHRD_PI  4.18879020478639098462

/*
 * STREQ: string-equality macro. STRNULL: test for empty string.
 * Note that string.h should be included if these are used.
 */

#define streq(x,y) (strcmp((x), (y)) == 0)
#define strnull(x) (strcmp((x), "") == 0)

/*
 * ABS: returns the absolute value of its argument.
 * MAX: returns the argument with the highest value.
 * MIN: returns the argument with the lowest value.
 */

#define ABS(x)   (((x)<0)?-(x):(x))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

/*
 * Prototypes for misc. functions in libZeno.a.
 */

void *allocate(int);                    /* alloc, check for errors, zero    */

double cputime(void);                   /* returns CPU time in minutes      */

void error(string, ...);                /* complain about error and exit    */

void eprintf(string, ...);              /* print message to stderr          */

bool scanopt(string, string);           /* scan options for keyword         */

stream stropen(string, string);         /* arguments are much like fopen    */




double run_simulation(double T, char* filename);

#endif  /* ! _stdinc_h */