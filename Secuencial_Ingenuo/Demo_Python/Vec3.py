
import math
class Vec3:
    '''
    Class to represent vectors of three dimensions
    '''
    def __init__(self,x:float, y:float, z:float) -> None:
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
    def __mul__(self, other):
        '''Redefine multiplication vector to vector and vector to int or float'''
        if isinstance(other, Vec3):  # Vec3 dot product
            return (
                self.x * other.x
                + self.y * other.y
                + self.z * other.z
            )
        elif isinstance(other, (int, float)):  # Scalar multiplication
            return Vec3(
                self.x * other,
                self.y * other,
                self.z * other,
            )
        else:
            raise TypeError("operand must be Vec3, int, or float")

    def __truediv__(self, other):
        '''Redefine divition between vector and number'''
        if isinstance(other, (int, float)):
            return Vec3(
                self.x / other,
                self.y / other,
                self.z / other,
            )
        else:
            raise TypeError("operand must be int or float")
    def norm(self):
        '''Get the normal of the vector'''
        mag = self.mag()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)
    def mag(self):
        '''Get the maginitude of the vector'''
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
      
    def __getitem__(self, item):
        '''Get x,y,z by index'''
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise IndexError("There are only three elements in the vector")
        
    def __repr__(self):
        'Representation of the vector'
        return f"Vector({self.x}, {self.y}, {self.z})"  
    
    def __str__(self) -> str:
        '''Pretty print'''
        return f"[{self.x}, {self.y}, {self.z}]"
