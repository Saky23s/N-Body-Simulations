/** 
 * @file treeload.c
 * @copyright (c) 1999 by Joshua E. Barnes, Tokyo, JAPAN
 * 
 * Routines to create tree.
 * 
 * Modifies the original work of Joshua E. Barnes to remove features
 * that are not required for this investigation 
 * 
 * Added free of memory
 * 
 * @author (modifications) Santiago Salas santiago.salas@estudiante.uam.es             
 **/

#include "../inc/stdinc.h"
#include "../inc/mathfns.h"
#include "../inc/vectmath.h"
#include "../inc/treedefs.h"

//Internal helpers
local int newtree(void);
local cellptr makecell(void);
local int expandbox(bodyptr, int);
local int loadbody(bodyptr);
local int subindex(bodyptr, cellptr);
local int hackcofm(cellptr, real, int);
local int setrcrit(cellptr, vector, real);
local int threadtree(nodeptr, nodeptr);
local nodeptr freecell = NULL;

//Max height of the tree
#define MAXLEVEL 32 
//Count cells by level
local int cellhist[MAXLEVEL];
//Count subnodes by level
local int subnhist[MAXLEVEL];

/**
 * Main funtion of three construction
 * 
 * It initializes the tree structure for hierarchical force calculation.
 * @param btab (bodyptr): Pointer to array of bodies
 * @param int (nbody): number of bodies
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
int maketree(bodyptr btab, int nbody)
{
    double cpustart;
    bodyptr p;
    int i;

    //Record time at start
    cpustart = cputime();

    //Flush existing tree or allocate memory 
    if(newtree() == STATUS_ERROR)
        return STATUS_ERROR;

    //Allocate the root cell
    root = makecell();
    if(root == NULL)
        return STATUS_ERROR;
        
    //Initialize the midpoint 
    CLRV(Pos(root));

    //And expand cell to fit
    if(expandbox(btab, nbody) == STATUS_ERROR)
        return STATUS_ERROR;

    //Insert all bodies into the tree
    for (p = btab; p < btab+nbody; p++)
    {
        if(loadbody(p) == STATUS_ERROR)
            return STATUS_ERROR;
    }

    //Init depth to 0
    tdepth = 0;     
    //Init tree histograms
    for (i = 0; i < MAXLEVEL; i++)
    {
        cellhist[i] = subnhist[i] = 0;
    }

    //Find center-of-mass coordinates
    if(hackcofm(root, rsize, 0) == STATUS_ERROR)
        return STATUS_ERROR;

    //Add next and more links
    if(threadtree((nodeptr) root, NULL) == STATUS_ERROR)
        return STATUS_ERROR; 

    //Store elapsed CPU time
    cputree = cputime() - cpustart;

    return STATUS_OK;
}

local int newtree(void)
/**
 * Fuctions that reclaims cells in a tree (sets them as free) 
 * as preparation to build a new one
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    static bool firstcall = TRUE;
    nodeptr p;

    //If there already are cells to reclaim
    if (!firstcall) 
    {   
        //Starting with the root
        p = (nodeptr) root; 
        //Scan tree
        while (p != NULL) 
        {
            //If we found a cell                     
            if (Type(p) == CELL) 
            {    
                //Save existing list and add it to the front   
                Next(p) = freecell;
                freecell = p;
                p = More(p);
            }
            //If we found a body skip it
            else
            {
                p = Next(p);
            }
        }
    }
    //Nothing to reclaim
    else
    {
        firstcall = FALSE;
    }

    //Flush existing tree
    root = NULL;                                

    return STATUS_OK;                               
}

local cellptr makecell(void)
/**
 * Funtion that returns a pointer to a free cell
 * This free cell can be one that was marked as free by newtree or a newly allocated one
 * 
 * @return c (cellptr): Pointer to free cell 
 */
{
    cellptr c;
    int i;

    //If there is no free cells left allocate a new one
    if (freecell == NULL)  
    {
        c = (cellptr) allocate(sizeof(cell)); 
        if(c == NULL)
            return NULL;
    }
    //Else take the free cell in front of the list
    else 
    {
        c = (cellptr) freecell;
        freecell = Next(c);
    }

    //Initialize node
    Type(c) = CELL; 
    Update(c) = FALSE;

    //Set subcells to empty
    for (i = 0; i < NSUB; i++) 
    {
        Subp(c)[i] = NULL;                      
    } 

    //Count one more cell
    ncell++;
    
    //Return pointer to cell
    return c;
}

local int expandbox(bodyptr btab, int nbody)
/**
 * Funtion to find range of coordinate values (with respect to root)
 * and expand root cell to fit.  The size is doubled at each step to
 * take advantage of exact representation of powers of two.
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    real dmax, d;
    bodyptr p;
    int k;

    //Variable to keep track of max value
    dmax = 0.0;

    //Loop over all bodies and dimensions
    for (p = btab; p < btab+nbody; p++) 
    {
        for (k = 0; k < NDIM; k++) 
        {       
            //Find distance to midpoint and if its the biggest yet store it        
            d = rabs(Pos(p)[k] - Pos(root)[k]); 
            if (d > dmax)
                dmax = d;
        }
    }

    //Loop until a value fits by doubling the box value each time
    while (rsize < 2 * dmax) 
    {
        rsize = 2 * rsize;
    }    
    return STATUS_OK;              
}

local int loadbody(bodyptr p)
/**
 * Function to descend into the tree and insert body p in appropriate place.
 * @param p (bodyptr): Pointer to body to be inserted
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    cellptr q, c;
    int qind, k;
    real qsize, dist2;
    vector distv;

    //Starting with the root
    q = root;
    //Get subsell
    qind = subindex(p, q); 
    //Cell size
    qsize = rsize;

    //Loop into descending tree
    while (Subp(q)[qind] != NULL) 
    {   
        //If we reach a body
        if (Type(Subp(q)[qind]) == BODY)
        {     
            //Check that positions differ
            DOTPSUBV(dist2, distv, Pos(p), Pos(Subp(q)[qind]));
            if (dist2 == 0.0)   
            {
                error("loadbody: two bodies have same position\n");
                return STATUS_ERROR;
            }
            //Allocate a new cell
            c = makecell();
            if(c == NULL)
                return STATUS_ERROR;

            //Init mid point and offset from parent
            for (k = 0; k < NDIM; k++)  
            {
                Pos(c)[k] = Pos(q)[k] + (Pos(p)[k] < Pos(q)[k] ? - qsize : qsize) / 4;

            }
            //Put the body in the cell, and link cell to tree
            Subp(c)[subindex((bodyptr) Subp(q)[qind], c)] = Subp(q)[qind];
            Subp(q)[qind] = (nodeptr) c;
        }
        //Advance to the next level
        q = (cellptr) Subp(q)[qind]; 
        //get next index
        qind = subindex(p, q);
        //Shrink current cell
        qsize = qsize / 2;
    }
    
    //Found place for body, store it
    Subp(q)[qind] = (nodeptr) p;
    return STATUS_OK;
}

local int subindex(bodyptr p, cellptr q)
/**
 * Function to compute subcell index for body p in cell q.
 * 
 * @param p (bodyptr): Pointer to the body we are calculating index
 * @param q (cellptr): Pointer to cell that contains the body
 * 
 * @return ind (int): Subcell index of body p in cell q
 */
{
    int ind, k;

    //Accumulate subcell index
    ind = 0;
    
    //Loop over dimensions
    for (k = 0; k < NDIM; k++)
    {   
        //If beyond midpoint slip over subcells
        if (Pos(q)[k] <= Pos(p)[k])
            ind += NSUB >> (k + 1);
    }
    return ind;
}

local int hackcofm(cellptr p, real psize, int lev)
/**
 * Funtion that decends into the tree to find center-of-mass coordinates and 
 * set critical cell radius
 * 
 * This funtion works recursivly, finding the center of mass for cell p will also find it
 * for all subcells of p
 * @param p (cellptr): Cell which center of mass we want to find
 * @param psize (real): Size of the cell
 * @param lev (int): Depth of the cell
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    vector cmpos, tmpv;
    int i, k;
    nodeptr q;
    real dpq;

    //Remember maximum level
    tdepth = MAX(tdepth, lev);
    //Count cells by level
    cellhist[lev]++;
    //Init cell total mass
    Mass(p) = 0.0; 
    //Center mass position
    CLRV(cmpos);
    //Loop over subnodes
    for (i = 0; i < NSUB; i++) 
    {   
        //Skip null subnodes
        if ((q = Subp(p)[i]) != NULL) 
        {   
            //Count existing subnodes
            subnhist[lev]++;
            
            //If node type is cell do the same for that subnode
            if (Type(q) == CELL)
            {
                hackcofm((cellptr) q, psize/2, lev+1);
            }
            
            //Propagate update request 
            Update(p) |= Update(q);
            //Accumulate total mass
            Mass(p) += Mass(q); 
            //Weight position by mass
            MULVS(tmpv, Pos(q), Mass(q));
            //And add sum to center-of-mass position
            ADDV(cmpos, cmpos, tmpv); 
        }
    }   
    //If cell has mass find center of mass position             
    if (Mass(p) > 0.0) 
    {                        
        DIVVS(cmpos, cmpos, Mass(p));
    } 
    //But if there is no mass inside use geo. center for now
    else 
    { 
        SETV(cmpos, Pos(p));
    }

    //Check center of mass of cell
    for (k = 0; k < NDIM; k++)
    {   
        //If its actually outside the cell
        if (cmpos[k] < Pos(p)[k] - psize/2 || Pos(p)[k] + psize/2 <= cmpos[k])
        {
            error("hackcofm: tree structure error\n");
            return STATUS_ERROR;
        }
    }  

    //Set critical radius
    if(setrcrit(p, cmpos, psize) == STATUS_ERROR)
        return STATUS_ERROR; 

    //Set center of mass position
    SETV(Pos(p), cmpos);

    return STATUS_OK;
}

local int setrcrit(cellptr p, vector cmpos, real psize)
/**
 * Funtion that assigns critical radius for cell p, using center-of-mass
 * position cmpos and cell size psize.
 * 
 * @param p (cellptr): Cell which we want to set the critical radius
 * @param cmpos (vector): Center of mass position
 * @param psize (real): size of the cell
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    real bmax2, d;
    int k;

    //If exact calculations are used the set the critia; radius as open
    if (theta == 0.0)
        Rcrit2(p) = rsqr(2 * rsize);
    
    //If not exact
    else 
    {   
        //FInd offset from center
        DISTV(d, cmpos, Pos(p));
        //Use size plus offset
        Rcrit2(p) = rsqr(psize / theta + d);
    }
    return STATUS_OK;
}

local int threadtree(nodeptr p, nodeptr n)
/**
 * Funtion that does a recursive treewalk starting from node p,
 * with next stop n, installing Next and More links.
 * 
 * @param p (nodeptr): Starting node
 * @param n (nodeptr): Next stop
 * 
 * @return status (int): STATUS_ERROR (0) in case of error STATUS_OK(1) in case everything when ok
 */
{
    int ndesc, i;
    nodeptr desc[NSUB+1];

    //Set link to next node
    Next(p) = n;
    
    //If descendents to thread
    if (Type(p) == CELL) 
    {
        ndesc = 0;

        //Count occupied subcells and store it in the table
        for (i = 0; i < NSUB; i++)
        {
            if (Subp(p)[i] != NULL)
                desc[ndesc++] = Subp(p)[i];
        }

        //Link more as the first one on the table
        More(p) = desc[0];
        //And thread last one to next
        desc[ndesc] = n;

        //Loop over descendants and thread them together
        for (i = 0; i < ndesc; i++)
        {
            if(threadtree(desc[i], desc[i+1]) == STATUS_ERROR)
                return STATUS_ERROR;
        }
    }
    return STATUS_OK;
}

void freetree(bodyptr btab)
/**
 * Funtion to free allocated memory
 * @param btab (bodyptr): Pointer to array of bodies
 */
{
    nodeptr p;
    nodeptr q;

    //Starting with the root
    p = (nodeptr) root; 
    //Scan tree
    while (p != NULL) 
    {
        //If we found a cell                     
        if (Type(p) == CELL) 
        {    
            //Save existing list and add it to the front   
            Next(p) = freecell;
            freecell = p;
            p = More(p);

        }
        //If we found a free it and skip it
        else
        {
            p = Next(p);
        }
    }

    
    while((p = (nodeptr) freecell) != NULL)
    {
        freecell = Next(p);
        free((cellptr) p);
    }

    free(btab);
}