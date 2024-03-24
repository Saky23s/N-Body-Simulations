import numpy as np
import csv
import itertools
from Vec3 import Vec3

G = 6.674299999999999e-08 
dt = 3600

class Body:
    '''
    Class that defines a body in the system
    '''
    id_iter = itertools.count()
    def __init__(self, pos, mass, velocity) -> None:
        '''
        pos (Vec3): Starting position of the body
        mass (float): Mass of the body in Kg
        a (Vec3): Starting acceleration of the body
        '''
        #Id of the body for reference
        self.id = next(Body.id_iter)
        #Starting values
        self.pos = pos
        self.mass = mass
        self.velocity = velocity
        self.acceleration = Vec3(0.0,0.0,0.0)
    
    def accelerate(self, other):
        '''
        Update the position and potencial of the body

        forces (list(Vec3)): List of all forces between the body and other bodies in the system
        '''
        distance =  self.pos - other.pos
        distance_magnitud =  distance.mag()
        print(distance_magnitud)

        aceleration_magnitud = ((-G) * other.mass) / (distance_magnitud **3)
        aceleration =  distance * aceleration_magnitud
        self.acceleration = self.acceleration  + aceleration

    def move(self):
        print( self.acceleration, dt, self.acceleration * dt )
        print(self)
        self.velocity = self.velocity + self.acceleration * dt
        self.pos = self.pos + self.velocity * dt
        self.acceleration = Vec3(0.0,0.0,0.0)
        print(self)

    def __str__(self) -> str:
        '''
        Pretty print
        '''
        return f"Id: {self.id}\nPosition: {self.pos}\nMass: {self.mass}\nVelocity: {self.velocity}\n"
    

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
    forces = np.ndarray(shape=(n), dtype=Vec3)
    for i in range(n):
        forces[i] = Vec3(0.0,0.0,0.0)
    
    #Simulate 100 times
    t = 0.0
    for count in range(1000):
        #Calculate forces
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                bodies[i].accelerate(bodies[j])

        t += dt

        f = open(f"data/{count + 1}.csv", "w")

        for i in range(n):
            bodies[i].move()
            f.write(f"{bodies[i].pos.x},{bodies[i].pos.y},{bodies[i].pos.z}\n")
        f.close()

        
        
