/** 
 * @file vectmath.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * include file for vector/matrix operations.
 * 
 * This document has been modified lightly to remove funtions not needed in this investigations
 * this includes all funtions for 2 dimensions
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/
#ifndef _vectmath_h
#define _vectmath_h

#define NDIM 3
typedef real vector[NDIM];
typedef real matrix[NDIM][NDIM];

/*
 * Vector operations.
 */

#define CLRV(v)                 /* CLeaR Vector */                      \
{                                                                       \
    int _i;                                                             \
    for (_i = 0; _i < NDIM; _i++)                                       \
        (v)[_i] = 0.0;                                                  \
}

#define SETV(v,u)               /* SET Vector */                        \
{                                                                       \
    int _i;                                                             \
    for (_i = 0; _i < NDIM; _i++)                                       \
        (v)[_i] = (u)[_i];                                              \
}

#define ADDV(v,u,w)             /* ADD Vector */                        \
{                                                                       \
    (v)[0] = (u)[0] + (w)[0];                                           \
    (v)[1] = (u)[1] + (w)[1];                                           \
    (v)[2] = (u)[2] + (w)[2];                                           \
}

#define SUBV(v,u,w)             /* SUBtract Vector */                   \
{                                                                       \
    (v)[0] = (u)[0] - (w)[0];                                           \
    (v)[1] = (u)[1] - (w)[1];                                           \
    (v)[2] = (u)[2] - (w)[2];                                           \
}

#define MULVS(v,u,s)            /* MULtiply Vector by Scalar */         \
{                                                                       \
    (v)[0] = (u)[0] * s;                                                \
    (v)[1] = (u)[1] * s;                                                \
    (v)[2] = (u)[2] * s;                                                \
}

#define DIVVS(v,u,s)            /* DIVide Vector by Scalar */           \
{                                                                       \
    int _i;                                                             \
    for (_i = 0; _i < NDIM; _i++)                                       \
        (v)[_i] = (u)[_i] / (s);                                        \
}

#define DOTVP(s,v,u)            /* DOT Vector Product */                \
{                                                                       \
    (s) = (v)[0]*(u)[0] + (v)[1]*(u)[1] + (v)[2]*(u)[2];                \
}

#define DISTV(s,u,v)            /* DISTance between Vectors */          \
{                                                                       \
    real _tmp;                                                          \
    int _i;                                                             \
    _tmp = 0.0;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        _tmp += ((u)[_i]-(v)[_i]) * ((u)[_i]-(v)[_i]);                  \
    (s) = rsqrt(_tmp);                                                  \
}

/*
 * Matrix operations.
 */

#define CLRM(p)                 /* CLeaR Matrix */                      \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = 0.0;                                          \
}

#define SETMI(p)                /* SET Matrix to Identity */            \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = (_i == _j ? 1.0 : 0.0);                       \
}

#define SETM(p,q)               /* SET Matrix */                        \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = (q)[_i][_j];                                  \
}

#define ADDM(p,q,r)             /* ADD Matrix */                        \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = (q)[_i][_j] + (r)[_i][_j];                    \
}

#define SUBM(p,q,r)             /* SUBtract Matrix */                   \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = (q)[_i][_j] - (r)[_i][_j];                    \
}

#define MULMS(p,q,s)            /* MULtiply Matrix by Scalar */         \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = (q)[_i][_j] * (s);                            \
}

#define OUTVP(p,v,u)            /* OUTer Vector Product */              \
{                                                                       \
    int _i, _j;                                                         \
    for (_i = 0; _i < NDIM; _i++)                                       \
        for (_j = 0; _j < NDIM; _j++)                                   \
            (p)[_i][_j] = (v)[_i] * (u)[_j];                            \
}

/*
 * Enhancements for tree codes.
 */

#define DOTPSUBV(s,v,u,w)       /* SUB Vectors, form DOT Prod */        \
{                                                                       \
    (v)[0] = (u)[0] - (w)[0];    (s)  = (v)[0] * (v)[0];                \
    (v)[1] = (u)[1] - (w)[1];    (s) += (v)[1] * (v)[1];                \
    (v)[2] = (u)[2] - (w)[2];    (s) += (v)[2] * (v)[2];                \
}

#define DOTPMULMV(s,v,p,u)      /* MUL Mat by Vect, form DOT Prod */    \
{                                                                       \
    DOTVP(v[0], p[0], u);    (s)  = (v)[0] * (u)[0];                    \
    DOTVP(v[1], p[1], u);    (s) += (v)[1] * (u)[1];                    \
    DOTVP(v[2], p[2], u);    (s) += (v)[2] * (u)[2];                    \
}

#define ADDMULVS(v,u,s)         /* MUL Vect by Scalar, ADD to vect */   \
{                                                                       \
    (v)[0] += (u)[0] * (s);                                             \
    (v)[1] += (u)[1] * (s);                                             \
    (v)[2] += (u)[2] * (s);                                             \
}

#define ADDMULVS2(v,u,s,w,r)    /* 2 times MUL V by S, ADD to vect */   \
{                                                                       \
    (v)[0] += (u)[0] * (s) + (w)[0] * (r);                              \
    (v)[1] += (u)[1] * (s) + (w)[1] * (r);                              \
    (v)[2] += (u)[2] * (s) + (w)[2] * (r);                              \
}

#endif  /* ! _vectmath_h */
