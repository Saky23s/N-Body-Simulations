/** 
 * @file stdinc.h
 * @copyright (c) 1999 by Joshua E. Barnes, Tokyo, JAPAN. 
 * 
 * Standard include file
 * 
 * This document has been modified lightly to remove funtions not needed in this investigations
 * and to adapt it to work with our existing framework
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#ifndef _stdinc_h
#define _stdinc_h

//Always include stdio.h and stdlib.h.
#include <stdio.h>
#include <stdlib.h>

/***************************************/
//Local synonym for static declares an object as local to a source file.
#define local static

/***************************************/

//Define booleans 
typedef short int bool;

#if !defined(TRUE)
#define TRUE  ((bool) 1)
#define FALSE ((bool) 0)
#endif

//error checking
#define STATUS_ERROR 0
#define STATUS_OK 1

/***************************************/
//Define real
typedef double real, *realptr;

/***************************************/
//Returns the absolute value of its argument.
#define ABS(x)   (((x)<0)?-(x):(x))
//Returns the argument with the highest value.
#define MAX(a,b) (((a)>(b))?(a):(b))
//Returns the argument with the lowest value.
#define MIN(a,b) (((a)<(b))?(a):(b))

/***************************************/
//Prototypes for misc. functions in libZeno.a.
double cputime(void); 
void error(char*, ...); 

/***************************************/
//Main funtion of the simulation, creates, runs and frees a simulation
double run_simulation(double T, char* filename);
#endif  /* ! _stdinc_h */
