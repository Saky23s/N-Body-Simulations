import argparse
import csv
import sys
import time
import numpy as np
import struct

G = 1
dt = 0.01
speed = 0.05
softening = 0.1

class Simulation():
    def __init__(self, filepath) -> None:
        '''
        @param filepath (string): path to the starting configuration of the bodies, this file must be a csv
        '''
        self.bodies, self.masses, self.n = Simulation.load_bodies(filepath)

    
    def run(self, T):
        '''
        Funtion that will run the simulation for T internal seconds (this means that the ending positions of the bodies will be in time T)

        This funtion will calculate the positions of the bodies every in timesteps of 'dt'using the runge-kutta method
        and store them in data/ as csv files every 'speed' seconds
        
        @param T (float): Internal ending time of the simulation
        '''
        steps = int(T // dt)
        #Calculate the number of timesteps we must do before saving the data
        save_step = int(speed / dt)
        #Internal variables to measure time and keep track of csv files written
        clock_time = 0.0
        file_number = 1
        start_time = time.time()
        #Run simulation
        for step in range(steps):
            
            #Integrate next step using runge-kutta
            self.rk4()
            
            #Save data if we must
            if step % save_step == 0:
                with open(f"../../Graphics/data/{file_number}.bin", "wb") as f:
                    for i in range(self.n):
                        ioffset = i * 6
                        x = self.bodies[ioffset]
                        y = self.bodies[ioffset + 1]
                        z = self.bodies[ioffset + 2]
                        f.write(struct.pack('ddd', x, y, z))
                file_number += 1
            
            #Print information of time integrating
            clock_time += dt
            sys.stdout.flush()
            sys.stdout.write('Integrating: step = {} / {} | simulation time = {}'.format(step,steps,round(clock_time,3)) + '\r')
        #Print finishing time
        print('\n')
        print(f'Simulation completed in {time.time() - start_time} seconds')
    
    def rk4(self) -> None:
        '''
        Integrate the next timestep in the simulation using the Runga-Kutta method
        '''
        k1 = dt * self.calculate_acceleration(self.bodies) 
        k2 = dt * self.calculate_acceleration(self.bodies+0.5*k1)
        k3 = dt * self.calculate_acceleration(self.bodies+0.5*k2)
        k4 = dt * self.calculate_acceleration(self.bodies + k3)

        self.bodies = self.bodies + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)

    def calculate_acceleration(self, values:np.array):
        '''
        Funtion to calculate the velocity and acceleration of the bodies using the current positions and velocities

        @param values(np.array): an array of values order as x1,y1,z1,vx1,vz1,vz1,x2,y2,z2,vx2,vz2,vz2...xn,yn,zn,vxn,vzn,vzn
        @return (np.array): an array of values order as vx1,vy1,vz1,ax1,ay1,az1,vx2,vy2,vz2,ax2,ay2,az2...vxn,vyn,vzn,axn,ayn,azn
        '''
        k = np.zeros(values.size)
        for i in range(self.n):
            ioffset = i * 6 
            k[ioffset] = values[ioffset+3] #vx
            k[ioffset+1] = values[ioffset+4] #vy
            k[ioffset+2] = values[ioffset+5] #vz
            for j in range(self.n):
                #Make sure not to do the calculation for the same body
                if i == j:
                    continue
                joffset = j * 6 
                dx = values[joffset] - values[ioffset] #rx body 2 - rx body 1
                dy = values[joffset+1] - values[ioffset+1] #ry body 2 - ry body 1
                dz = values[joffset+2] - values[ioffset+2] #rz body 2 - rz body 1
                
                r = (dx**2+dy**2+dz**2+softening**2)**0.5 #distance magnitud with some softening
                
                ax = (G * self.masses[j] / r**3) * dx #Aceleration formula for x
                ay = (G * self.masses[j] / r**3) * dy #Aceleration formula for y
                az = (G * self.masses[j] / r**3) * dz #Aceleration formula for z
                k[ioffset+3] += ax 
                k[ioffset+4] += ay
                k[ioffset+5] += az      
        return k #returns vx,vy,vz,ax,ay,az for all of the bodies
    def load_bodies(filepath:str) -> list:
        '''
        Funtion to correctly load all body values to the simulation

        @param filepath (str): The path to the csv file that has the bodies. It must
        be in x,y,z,mass,vx,vy,vz ... and it must have a header

        @return [np.array(bodies), np.array(masses), len(masses)] (list): Returns a list
        the first value of the list is an array with all of the body positions and velocities following the Runge-Kutta-Standard
        the second value of the list is an array with all of the masses
        the third value is an int with the number of bodies
        '''
        bodies = list()
        masses = list()
        with open(filepath, "r") as file:
            csvreader = csv.reader(file)
            #Skip header
            next(csvreader)
            #Create bodies
            for row in csvreader:
                #Add x,y,z,vx,vy,vz
                bodies.extend([float(row[0]), float(row[1]), float(row[2]), float(row[4]), float(row[5]), float(row[6])])
                #Add mass
                masses.append(float(row[3]))
        return [np.array(bodies), np.array(masses), len(masses)]

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