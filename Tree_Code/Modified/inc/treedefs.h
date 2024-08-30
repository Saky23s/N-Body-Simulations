/** 
 * @file treedefs.h
 * @copyright (c) 1999 by Joshua E. Barnes, Tokyo, JAPAN. 
 * 
 * Include file for hierarchical force calculation routines.
 * These definitions are needed for treeload.c and treegrav.c, but this
 * file does not provide definitions for other parts of the N-body code. 
 * 
 * This document has been modified lightly to remove funtions not needed in this investigations
 * and to adapt it to work with our existing framework
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#ifndef _treedefs_h
#define _treedefs_h

/**
 * @struct node
 * @brief data common to BODY and CELL structures.
 */
typedef struct _node 
{   
    //Code for node type
    short type;
    //Status in force calculation
    bool update;
    //Total mass of node
    real mass;
    //Position of node
    vector pos;
    //Link to next in force calculation
    struct _node *next;
} node, *nodeptr;

//Easy access to structure atributes
#define Type(x)   (((nodeptr) (x))->type)
#define Update(x) (((nodeptr) (x))->update)
#define Mass(x)   (((nodeptr) (x))->mass)
#define Pos(x)    (((nodeptr) (x))->pos)
#define Next(x)   (((nodeptr) (x))->next)

//Type codes
#define BODY 01
#define CELL 02

/**
 * @struct body
 * @brief data structure used to represent particles.
 */
typedef struct 
{   
    //Data common to all nodes
    node bodynode;
    //Velocity of body
    vector vel;
    //Acceleration of body
    vector acc;
    //Potential at body
    real phi;
} body, *bodyptr;

//Easy access to structure atributes
#define Vel(x)    (((bodyptr) (x))->vel)
#define Acc(x)    (((bodyptr) (x))->acc)
#define Phi(x)    (((bodyptr) (x))->phi)


//Subcells per cell
#define NSUB (1 << NDIM)

/**
 * @struct cell
 * @brief structure used to represent internal nodes of tree.
 */
typedef struct 
{   
    //Data common to all nodes
    node cellnode; 
    //Critical center of mass radius^2
    real rcrit2;
    //Link to first descendent
    nodeptr more;
    union 
    {   
        //Descendents of cell
        nodeptr subp[NSUB];
        //Quad. moment of cell
        matrix quad;            
    } sorq;
} cell, *cellptr;

//Easy access to structure atributes
#define Rcrit2(x) (((cellptr) (x))->rcrit2)
#define More(x)   (((cellptr) (x))->more)
#define Subp(x)   (((cellptr) (x))->sorq.subp)
#define Quad(x)   (((cellptr) (x))->sorq.quad)

/*
 * GLOBAL: pseudo-keyword for storage class.
 */
#if !defined(global)
#  define global extern
#endif

/***************************************/
//Parameters for simulation
//force accuracy parameter
#define theta 1.0
//density smoothing parameter
#define softening 0.025
//Step size
#define dt 0.01
//Time until save
#define speed 0.1
//Density smoothing length
#define eps 0.025

/***************************************/
//Global funtions
//Funtion to construct tree
int maketree(bodyptr, int);
//Funtion to free tree memory
void freetree(bodyptr btab);
//Funtion to update forces on bodies
void gravcalc(void); 

/***************************************/
//Global variables
global cellptr root;                    /* pointer to root cell             */
global real rsize;                      /* side-length of root cell         */
global int ncell;                       /* count of cells in tree           */
global int tdepth;                      /* count of levels in tree          */
global real cputree;                    /* CPU time to build tree           */

global int actmax;                      /* maximum length of active list    */
global int nbbcalc;                     /* total body-body interactions     */
global int nbccalc;                     /* total body-cell interactions     */
global real cpuforce;                   /* CPU time for force calc          */

#endif /* ! _treedefs_h */
