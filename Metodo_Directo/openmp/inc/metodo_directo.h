#ifndef METODO_DIRECTO_H
#define METODO_DIRECTO_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "../../simulation.h"

/***************************************/
//Prototypes for I/O routines.
Simulation* load_bodies(char* filepath);
int output(Simulation* simulation, int* filenumber);
//Prototypes to free the memory.
void free_simulation(Simulation* simulation);

/***************************************/
//Prototypes for acceleration
int calculate_acceleration(Simulation* simulation);
#endif

