import math
import numpy as np
import csv
import itertools

G = 6.67e-11
dt = 60

class Body:
    '''
    Class that defines a body in the system
    '''
    id_iter = itertools.count()
    def __init__(self, pos, mass, p) -> None:
        '''
        pos (Vec3): Starting position of the body
        mass (float): Mass of the body in Kg
        p (Vec3): Starting potential of the body
        '''
        #Id of the body for reference
        self.id = next(Body.id_iter)
        #Starting values
        self.pos = pos
        self.mass = mass
        self.p = p

    def rAB(self, body2):
        '''
        Calculates the vector between two bodies

        body2 (Body): The second body

        return (Vec3): The vector between the two bodies
        '''
        return body2.pos - self.pos
    
    def fAB(self, body2):
        '''
        Calculates the vector between two bodies

        body2 (Body): The second body

        return (Vec3): The force vector
        '''
        norm = self.rAB(body2).norm()
        mag = self.rAB(body2).mag()
        holder = (-G*self.mass*body2.mass) / mag**2
        return Vec3(holder*norm.x, holder*norm.y, holder*norm.z)
    
    def update(self, forces):
        '''
        Update the position and potencial of the body

        forces (list(Vec3)): List of all forces between the body and other bodies in the system
        '''
        self.p = self.p + Vec3.aggregate(forces).scalarmul(dt)
        self.pos=self.pos+(self.p.scalarmul(dt/self.mass))

    def __str__(self) -> str:
        '''
        Pretty print
        '''
        return f"Id: {self.id}\nPosition: {self.pos}\nMass: {self.mass}\nPotential: {self.p}\n"
    

class Vec3:
    '''
    Class to represent vectors of three dimensions
    '''
    def __init__(self,x, y, z) -> None:
        '''
        x (float)
        y (float)
        z (float)
        '''
        self.x = x
        self.y = y
        self.z = z  
        
    def __add__(self, other):
        '''Redefine adition between two vectors vec1 + vec2'''
        return Vec3(other.x + self.x, other.y +self.y, other.z + self.z)
    def __pos__(self):
        '''Redefine positive vector +vec1'''
        return self
    def __sub__(self, other):
        '''Redefine subsrtaction between two vectoes vec1 - vec2'''
        return Vec3(self.x -  other.x, self.y - other.y, self.z -  other.z)
    def __neg__(self):
        '''Redefine negative vector'''
        return Vec3(-self.x, -self.y, -self.z)
    def norm(self):
        '''Get the normal of the vector'''
        mag = self.mag()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)
    def mag(self):
        '''Get the maginitude of the vector'''
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    def scalarmul(self, scalar):
        '''Multiply vector by scalar'''
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    def scalardiv(self, scalar):
        '''Divide vector by scalar'''
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    def aggregate(values):
        '''Add a bunch of vectors together'''
        start = Vec3(0.0, 0.0, 0.0)
        for i in values:
            start = start + i
        return start
    def __str__(self) -> str:
        '''Pretty print'''
        return f"[{self.x}, {self.y}, {self.z}]"

def load_bodies() -> list:
    '''Funtion to load all bodies
    
    Return (list(Body)): List of all bodies defined in data_python/bodies.csv
    '''
    result = list()
    #Open file
    with open("data_python/bodies.csv", "r") as file:
        csvreader = csv.reader(file)
        #Skip header
        headers = next(csvreader)
        #Create bodies
        for row in csvreader:
            result.append(Body(Vec3(float(row[0]), float(row[1]), float(row[2])), float(row[3]), Vec3(float(row[4]), float(row[5]), float(row[6]))))
    #Return list of bodies
    return result
if __name__ == "__main__":
    #Load all bodies
    bodies = load_bodies()
    #Array with all of the forces
    n = len(bodies)
    forces = np.ndarray(shape=(n,n), dtype=Vec3)
    #Simulate forever
    while(True):
        #Calculate forces
        for i in range(n):
            for j in range(i, n):
                #Fill diagonal with force 0
                if(i == j):
                    forces[i][j] = Vec3(0.0, 0.0, 0.0)
                    continue
                #Calculate the forces between two bodies
                force = bodies[i].fAB(bodies[j])
                forces[i][j] = -force
                forces[j][i] = force    

        #Update bodies
        for i in range(n):
            bodies[i].update(forces[i, :])
            print(bodies[i])
