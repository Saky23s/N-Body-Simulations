/** 
 * @file treegrav.c
 * @copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i. 
 * 
 * Routines to compute gravity.
 * 
 * Modifies the original work of Joshua E. Barnes to remove features
 * that are not required for this investigation 
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../inc/treedefs.h"

//Internal helpes
local int walktree(nodeptr *, nodeptr *, cellptr, cellptr,
                    nodeptr, real, vector);
local bool accept(nodeptr, real, vector);
local int walksub(nodeptr *, nodeptr *, cellptr, cellptr,
                   nodeptr, real, vector);
local int gravsum(bodyptr, cellptr, cellptr);
local int sumnode(cellptr, cellptr, vector, real *, vector);
local int sumcell(cellptr, cellptr, vector, real *, vector);

//Lists of active nodes and interactions.
//Active list fudge factor
#if !defined(FACTIVE)
#  define FACTIVE  0.75 
#endif

//Length as allocated 
local int actlen;

//Active pointer addresses an array of node pointers which will be examined when constructing interaction lists.
local nodeptr *active;

//Interact pointer addresses a linear array of cells which list all the interactions acting on a body
local cellptr interact;


int gravcalc(void)
/**
 * This funtion calculates the forces of all particles with a single recursive scan of the tree
 * 
 * The tree structure to be used by gravcalc is addressed by the global root pointer; also referenced are the tree depth tdepth and root cell size rsize.
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    
    vector rmid;

    //Estimate list length
    actlen = FACTIVE * 400 * tdepth; 

    //Allow for opening angle
    actlen = actlen * rpow(theta, -2.5); 

    //Allocate temporal storage 
    active = (nodeptr *) calloc(actlen * sizeof(nodeptr), 1);
    interact = (cellptr) calloc(actlen * sizeof(cell), 1);
    if(active == NULL || interact == NULL)
        return STATUS_ERROR;


    //Init active list
    active[0] = (nodeptr) root;

    //Set center of root cell
    CLRV(rmid);

    //Scan tree and update forces
    if (walktree(active, active + 1, interact, interact + actlen, (nodeptr) root, rsize, rmid) == STATUS_ERROR)
    {
        free(active);
        free(interact);
        return STATUS_ERROR;
    }
    free(active);
    free(interact);
    return STATUS_OK;
}

local int walktree(nodeptr *aptr, nodeptr *nptr, cellptr cptr, cellptr bptr, nodeptr p, real psize, vector pmid)
/**
 * This funtion computes gravity on bodies within node p. This is accomplished via a recursive scan of p and its descendents. 
 * At each point in the scan, information from levels between the root and p is contained in a set of nodes which will appear on the final interaction lists of all bodies within p
 *
 * @param aptr (nodeptr*): List of active node start 
 * @param nptr (nodeptr*): List of active node end
 * @param cptr (cellptr): Cell array pointer
 * @param bptr (cellptr): Body array pointer
 * @param p (nodeptr): Node which force we are calculating
 * @param psize (real): linear size of p
 * @param pmid (vector) geometic midpoint of p
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    nodeptr *np, *ap, q;
    int actsafe;

    //Are new forces needed?
    if (Update(p)) 
    {    
        //Start a new active list                        
        np = nptr;
        //Leave room for NSUB more
        actsafe = actlen - NSUB;

        //Loop over active nodes
        for (ap = aptr; ap < nptr; ap++)
        {   
            //If node is a cell
            if (Type(*ap) == CELL) 
            {   
                //If it passes the test copy interaction list and bump cell array ptr
                if (accept(*ap, psize, pmid)) 
                {   
                    
                    Mass(cptr) = Mass(*ap);
                    SETV(Pos(cptr), Pos(*ap));
                    SETM(Quad(cptr), Quad(*ap));
                    cptr++;
                } 
                //If it does not passes the test
                else 
                {   
                    //Check if list has room
                    if (np - active >= actsafe)
                    {
                        printf("walktree: active list overflow\n");
                        return STATUS_ERROR;
                    }
                    
                    ///Loop over all subcells and put them on new active list
                    for (q = More(*ap); q != Next(*ap); q = Next(q))
                    {
                        *np++= q;
                    }
                }
            } 
            //Node is a body
            else 
            {   
                //if not self-interaction
                if (*ap != p) 
                {
                    //Bump body array ptr
                    --bptr;
                    //Copy data to array
                    Mass(bptr) = Mass(*ap);
                    SETV(Pos(bptr), Pos(*ap));
                }
            } 
        }
        
        //If new actives listed then visit next level
        if (np != nptr)
        {
            if(walksub(nptr, np, cptr, bptr, p, psize, pmid) == STATUS_ERROR)
                return STATUS_ERROR;
        }
        //If no actives left means we must have found a body
        else 
        {
            if (Type(p) != BODY)
            {
                printf("walktree: recursion terminated with cell\n");
                return STATUS_ERROR;
            }
            if(gravsum((bodyptr) p, cptr, bptr) == STATUS_ERROR)
                return STATUS_ERROR;
        }
    }

    return STATUS_OK;
}

local bool accept(nodeptr c, real psize, vector pmid)
/**
 * This funtion checks if a cell critical radius does not intersect cell p 
 * 
 * @param c (nodeptr): Cell we are checking
 * @param psize (real): linear size of p
 * @param pmid (vector) geometic midpoint of p
 * 
 * @return bool, TRUE(1) if cell is accepted meaning that it does not intersect or FALSE(0) otherwise
 */
{
    real dmax, dsq, dk;
    int k;

    //Maximum distance
    dmax = psize;
    //Min distance
    dsq = 0.0;
    //Loop over the 3 dimensions
    for (k = 0; k < NDIM; k++) 
    {       
        //Absolute value of distance to midpoint 
        dk = Pos(c)[k] - pmid[k];
        if (dk < 0)
            dk = - dk;
        
        //Keep track of max value
        if (dk > dmax)
            dmax = dk;
        
        //Allow for size of cell 
        dk -= ((real) 0.5) * psize;

        //Sum min dist to cell ^2
        if (dk > 0)
            dsq += dk * dk;                     
    }

    //Test angular criterion and adjancency criterion
    return (dsq > Rcrit2(c) && dmax > ((real) 1.5) * psize);
}

local int walksub(nodeptr *nptr, nodeptr *np, cellptr cptr, cellptr bptr, nodeptr p, real psize, vector pmid)
/**
 * Test next level's active list against subnodes of p.
 * 
 * All parameters have the same value as walktree
 * @param aptr (nodeptr*): List of active node start 
 * @param nptr (nodeptr*): List of active node end
 * @param cptr (cellptr): Cell array pointer
 * @param bptr (cellptr): Body array pointer
 * @param p (nodeptr): Node which force we are calculating
 * @param psize (real): linear size of p
 * @param pmid (vector) geometic midpoint of p
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    real poff;
    nodeptr q;
    int k;
    vector nmid;

    //Mid ofsset
    poff = psize / 4;

    //If the node is a cell, fanout over descendents
    if (Type(p) == CELL) 
    {
        for (q = More(p); q != Next(p); q = Next(q)) 
        {   
            //Locate each's midpoint
            for (k = 0; k < NDIM; k++)   
            {
                nmid[k] = pmid[k] + (Pos(q)[k] < pmid[k] ? - poff : poff);
            } 

            //Invoke walktree for each descendent, with the appropriately shifted cell center. 
            if(walktree(nptr, np, cptr, bptr, q, psize / 2, nmid) == STATUS_ERROR)
                return STATUS_ERROR;
        }
    } 
    else 
    {   
        //locate next midpoint
        for (k = 0; k < NDIM; k++)
        {
            nmid[k] = pmid[k] + (Pos(p)[k] < pmid[k] ? - poff : poff);
        }
        
        //Continuing the search to the next level by `virtually' extending the tree.
        if(walktree(nptr, np, cptr, bptr, p, psize / 2, nmid) == STATUS_ERROR)
            return STATUS_ERROR;
    }

    return STATUS_OK;
}

local int gravsum(bodyptr p0, cellptr cptr, cellptr bptr)
/**
 * This funtion computes gravitational field at body p0
 * 
 * @param p0 (bodyptr): The body requiring updated forces 
 * @param cptr (cellptr): pointer to cell interaction list
 * @param bptr (cellptr): pointer to body interaction list
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    vector pos0, acc0;
    real phi0;

    //Copy position of body
    SETV(pos0, Pos(p0));
    //Init total potential and total acceleration
    phi0 = 0.0; 
    CLRV(acc0);

    // Not using quad moments
    if(sumnode(interact, cptr, pos0, &phi0, acc0) == STATUS_ERROR)
        return STATUS_ERROR;
    
    // sum cell forces wo quads
    if(sumnode(bptr, interact + actlen, pos0, &phi0, acc0) == STATUS_ERROR)
        return STATUS_ERROR;
    //sum forces from bodies 
    //Store total potential and acceleration
    Phi(p0) = phi0;
    SETV(Acc(p0), acc0);

    return STATUS_OK;
}

local int sumnode(cellptr start, cellptr finish, vector pos0, real *phi0, vector acc0)
/**
 * This funtion sums up interactions without quadrupole corrections.
 * 
 * @param start (cellptr): Front of the interaction list
 * @param finish (cellptr): Back of the interaction list
 * @param pos0 (vector): Place where the force is being evaluated
 * @param phi0 (real*): Resulting potential
 * @param acc0 (vector): Resulting acceleration
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    cellptr p;
    real eps2, dr2, drab, phi_p, mr3i;
    vector dr;

    //Avoid extra multiplys
    eps2 = softening * softening;

    //Loop over node list
    for (p = start; p < finish; p++) 
    {   
        //Compute separation and distance squared
        DOTPSUBV(dr2, dr, Pos(p), pos0);

        //Add softening
        dr2 += eps2;

        //Form scalar distance
        drab = rsqrt(dr2);

        //Get partial potential
        phi_p = Mass(p) / drab;

        //Decrement total potential
        *phi0 -= phi_p;

        //Form scale factor for dr
        mr3i = phi_p / dr2; 

        //Sum partial acceleration
        ADDMULVS(acc0, dr, mr3i);
    }

    return STATUS_OK;
}

/*
 * SUMCELL: add up body-cell interactions.
 */

local int sumcell(cellptr start, cellptr finish, vector pos0, real *phi0, vector acc0)
/**
 * This routine is similar to sumnode, but includes quadrupole-moment corrections (Hernquist 1987) to improve the forces and potentials generated by cells.
 * 
 * @param start (cellptr): Front of the interaction list
 * @param finish (cellptr): Back of the interaction list
 * @param pos0 (vector): Place where the force is being evaluated
 * @param phi0 (real*): Resulting potential
 * @param acc0 (vector): Resulting acceleration
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */                   
{
    cellptr p;
    real eps2, dr2, drab, phi_p, mr3i, drqdr, dr5i, phi_q;
    vector dr, qdr;

    //Avoid extra multiplys
    eps2 = softening * softening;
    
    //Loop over node list
    for (p = start; p < finish; p++) 
    {   
        //Do mono part of force
        DOTPSUBV(dr2, dr, Pos(p), pos0);
        dr2 += eps2;
        drab = rsqrt(dr2);
        phi_p = Mass(p) / drab;
        mr3i = phi_p / dr2;

        //Do quad part of force
        DOTPMULMV(drqdr, qdr, Quad(p), dr);
        dr5i = ((real) 1.0) / (dr2 * dr2 * drab);
        phi_q = ((real) 0.5) * dr5i * drqdr;

        //Add mono and quad pot
        *phi0 -= phi_p + phi_q;                 
        mr3i += ((real) 5.0) * phi_q / dr2;

        //Add mono and quad acc
        ADDMULVS2(acc0, dr, mr3i, qdr, -dr5i); 
    }
    return STATUS_OK;
}
