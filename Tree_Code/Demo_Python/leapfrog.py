
class Node:
    def __init__(self, type, update, mass, pos, next) -> None:
        self.type = type
        self.update = update
        self.mass = mass
        self.pos = pos
        self.next = next

class Body:
    def __init__(self, bodynode, vel, acc, phi):
        self.bodynode = bodynode
        self.vel = vel
        self.acc = acc
        self.phi = phi


class Sorq:
    def __init__(self) -> None:
        self.subp = list()
        self.quad = list()

class Cell:
    def __init__(self, cellnode, rcrit2, more, sorq) -> None:
        self.cellnode = cellnode
        self.rcrit2 = rcrit2
        self.more = more
        self.sorq = sorq
        
    def __init__(self):
        self.cellnode = None
        self.rcrit2 = None
        self.more = None
        self.sorq = None
 
root = Cell()

def maketree(btab: Body, nbody: int):
 
    newtree();                                  /* flush existing tree, etc */
    root = makecell();                          /* allocate the root cell   */
    CLRV(Pos(root));                            /* initialize the midpoint  */
    expandbox(btab, nbody);                     /* and expand cell to fit   */
    for (p = btab; p < btab+nbody; p++)         /* loop over all bodies     */
        loadbody(p);                            /* insert each into tree    */
    bh86 = scanopt(options, "bh86");            /* set flags for alternate  */
    sw94 = scanopt(options, "sw94");            /* ...cell opening criteria */
    if (bh86 && sw94)                           /* can't have both at once  */
        error("maketree: incompatible options bh86 and sw94\n");
    tdepth = 0;                                 /* init count of levels     */
    for (i = 0; i < MAXLEVEL; i++)              /* and init tree histograms */
        cellhist[i] = subnhist[i] = 0;
    hackcofm(root, rsize, 0);                   /* find c-of-m coords, etc  */
    threadtree((nodeptr) root, NULL);           /* add next and more links  */
    if (usequad)                                /* if including quad terms  */
        hackquad(root);                         /* find quadrupole moments  */
    cputree = cputime() - cpustart;             /* store elapsed CPU time   */
}


 
def newtree():

    firstcall = True
    if (firstcall == False):                    /* if cells to reclaim      */
        p = root;                     /* start with the root      */
        while (p != NULL)                       /* loop scanning tree       */
            if (Type(p) == CELL) {              /* if we found a cell to    */
                Next(p) = freecell;             /* then save existing list  */
                freecell = p;                   /* and add it to the front  */
                p = More(p);                    /* then scan down tree      */
            } else                              /* else, skip over bodies   */
                p = Next(p);                    /* by going on to the next  */
    } else                                      /* else nothing to reclaim  */
        firstcall = FALSE;                      /* so just note it          */
    root = NULL;                                /* flush existing tree      */
    ncell = 0;                                  /* reset cell count         */
}