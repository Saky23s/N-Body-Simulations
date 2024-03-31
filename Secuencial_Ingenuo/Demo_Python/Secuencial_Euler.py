import struct
import time
import numpy as np
import csv
import itertools
import argparse
import sys 

from Vec3 import Vec3

#Gravital constant ajusted to grams
#G = 6.674299999999999e-08 

#For most stating configurations online G = 1 is used
G = 1

#Timesteps in seconds
speed = 0.05
dt = 0.001

class Body:
    '''
    Class that defines a body in the system
    '''
    id_iter = itertools.count()
    def __init__(self, pos, mass, velocity) -> None:
        '''
        pos (Vec3): Starting position of the body
        mass (float): Mass of the body in grams
        velocity (Vec3): Starting velocity of the body in m/s
        '''
        #Id of the body for reference
        self.id = next(Body.id_iter)
        #Starting values
        self.pos = pos
        self.mass = mass
        self.velocity = velocity
        #Acceleration holder
        self.acceleration = Vec3(0.0,0.0,0.0)
    
    def accelerate(self, other):
        '''
        Calculate the acceleration generated on this body by the other body in the system

        other (Body): The other body which mass will acelerate ours
        '''
        #Calculate the distance between the two bodies
        distance =  other.pos - self.pos
        #Get the magnitud
        distance_magnitud =  distance.mag(0.1)
        #Calculate the acceleration using the formula a = -Gm/r³
        aceleration = distance * (G * other.mass) / (distance_magnitud ** 3)
        #Agregate acceleration
        self.acceleration = self.acceleration  + aceleration

    def move(self):
        '''
        Use the current aceleration to calculate new velocity and position of the body.
        Resets acceleration to 0,0,0
        '''
        self.velocity = self.velocity + self.acceleration * dt
        self.pos = self.pos + self.velocity * dt
        self.acceleration = Vec3(0.0,0.0,0.0)

    def __str__(self) -> str:
        '''
        Pretty print
        '''
        return f"Id: {self.id}\nPosition: {self.pos}\nMass: {self.mass}\nVelocity: {self.velocity}\Acceleration: {self.acceleration}\n"
    

class Simulation():
    '''
    Class to load all bodies and take care of doing the simulation
    '''
    def __init__(self, filepath) -> None:
        '''
        filepath (string): path to the starting configuration of the bodies, this file must be a csv
        '''
        self.bodies = load_bodies(filepath)
        self.n = len(self.bodies)

    def run(self, T):
        '''
        Funtion that will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)

        This funtion will calculate the positions of the bodies every in timesteps of 'dt' and store them in data/ as csv files every 'speed' seconds
        
        T (float): Internal ending time of the simulation
        '''
        #Calculate how many timesteps do we need to do
        steps = int(T // dt)
        #Calculate the number of timesteps we must do before saving the data
        save_step = int(speed / dt)
        #Internal variables to measure time and keep track of csv files written
        clock_time = 0.0
        file_number = 1
        start_time = time.time()
        #Run simulation
        for step in range(steps):
            #For all bodies
            for i in range(self.n):
                #For all other bodies
                for j in range(self.n):
                    #Not for yourself
                    if i == j:
                        continue
                    #Calculate acceleration made to body i by body j
                    self.bodies[i].accelerate(self.bodies[j])
            
            #Move bodies using the acceleration 
            for i in range(self.n):
                self.bodies[i].move()

            #Save data if we must            
            if step % save_step == 0:
                with open(f"../../Graphics/data/{file_number}.bin", "wb") as f:
                    for i in range(self.n):
                        f.write(struct.pack('ddd', self.bodies[i].pos.x, self.bodies[i].pos.y, self.bodies[i].pos.z))
                file_number += 1
            
            #Print information of time integrating
            clock_time += dt
            sys.stdout.flush()
            sys.stdout.write('Integrating: step = {} / {} | simulation time = {}'.format(step,steps,round(clock_time,3)) + '\r')
        #Print finishing time
        print('\n')
        print(f'Simulation completed in {time.time() - start_time} seconds')

def load_bodies(filepath) -> list:
    '''
    Helper funtion to load all values from a csv file in filepath

    filepath (string): the path to the file from which we want to open the data. The file must
    be a csv with a header x,y,z,mass,vx,vy,vz and then one row per body following the header structure
    '''
    result = list()
    #Open file
    with open(filepath, "r") as file:
        csvreader = csv.reader(file)
        #Skip header
        headers = next(csvreader)
        #Create bodies
        for row in csvreader:
            result.append(Body(Vec3(float(row[0]), float(row[1]), float(row[2])), float(row[3]), Vec3(float(row[4]), float(row[5]), float(row[6]))))
    #Return list of bodies
    return result

def argument_parser():
    '''Helper funtion to parse command line arguments arguments'''
    parser = argparse.ArgumentParser(description='Euler N-Body simulation')
    parser.add_argument('time', help='The ending time for the bodies in the simulation', type=float)
    parser.add_argument('filepath', help='File path to csv with starting data')
    return parser.parse_args()

if __name__ == "__main__":
    args = argument_parser()
    
    simulation = Simulation(args.filepath)
    simulation.run(args.time)
    

        
        
