/****************************************************************************/
/* TREEDEFS.H: include file for hierarchical force calculation routines.    */
/* These definitions are needed for treeload.c and treegrav.c, but this     */
/* file does not provide definitions for other parts of the N-body code.    */
/* Copyright (c) 1999 by Joshua E. Barnes, Tokyo, JAPAN.                    */
/****************************************************************************/

#ifndef _treedefs_h
#define _treedefs_h

/*
 * NODE: data common to BODY and CELL structures.
 */

typedef struct _node 
{
    short type;                 /* code for node type */
    bool update;                /* status in force calc */
    real mass;                  /* total mass of node */
    vector pos;                 /* position of node */
    struct _node *next;         /* link to next force calc */
} node, *nodeptr;

#define Type(x)   (((nodeptr) (x))->type)
#define Update(x) (((nodeptr) (x))->update)
#define Mass(x)   (((nodeptr) (x))->mass)
#define Pos(x)    (((nodeptr) (x))->pos)
#define Next(x)   (((nodeptr) (x))->next)

#define BODY 01                 /* type code for bodies */
#define CELL 02                 /* type code for cells */

/*
 * BODY: data structure used to represent particles.
 */

typedef struct 
{
    node bodynode;              /* data common to all nodes */
    vector vel;                 /* velocity of body */
    vector acc;                 /* acceleration of body */
    real phi;                   /* potential at body */
} body, *bodyptr;

#define Vel(x)    (((bodyptr) (x))->vel)
#define Acc(x)    (((bodyptr) (x))->acc)
#define Phi(x)    (((bodyptr) (x))->phi)

/*
 * CELL: structure used to represent internal nodes of tree.
 */

#define NSUB (1 << NDIM)        /* subcells per cell */

typedef struct 
{
    node cellnode;              /* data common to all nodes */
    real rcrit2;                /* critical c-of-m radius^2 */
    nodeptr more;               /* link to first descendent */
    union 
    {
        nodeptr subp[NSUB];     /* descendents of cell */
        matrix quad;            /* quad. moment of cell */
    } sorq;
} cell, *cellptr;

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

/*
 * Parameters for tree construction and force calculation.
 */


//force accuracy parameter
#define theta 1.0

//density smoothing parameter
#define softening 0.025

#define dt 0.01
#define speed 0.1


/*
 * Tree construction.
 */

int maketree(bodyptr, int);            /* construct tree structure         */

global cellptr root;                    /* pointer to root cell             */
global real rsize;                      /* side-length of root cell         */
global int ncell;                       /* count of cells in tree           */
global int tdepth;                      /* count of levels in tree          */
global real cputree;                    /* CPU time to build tree           */
global int nbody;

/*
 * Force calculation.
 */

void gravcalc(void);                    /* update force on bodies           */

global int actmax;                      /* maximum length of active list    */
global int nbbcalc;                     /* total body-body interactions     */
global int nbccalc;                     /* total body-cell interactions     */
global real cpuforce;                   /* CPU time for force calc          */


//error checking
#define STATUS_ERROR 0
#define STATUS_OK 1

#endif /* ! _treedefs_h */
