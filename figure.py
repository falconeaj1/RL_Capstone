import random
class Figure:
    # x,y are relative to game board field
    x = 0
    y = 0

    # EVERY property has empty spot in front so that 0 index (empty for board) doesn't get used
    
    names = ['', 'I', 'J', 'L', 'O', 'S', 'T', 'Z']
    figures = [
        [],
        [[4, 5, 6, 7], [2, 6, 10, 14], [8,9,10,11], [1,5,9,13]], # I, 
        [[0, 4, 5, 6], [1, 2, 5, 9], [4, 5, 6, 10], [1, 5, 9, 8]], # J
        [[4, 5, 6, 2], [1, 5, 9, 10], [8, 4, 5, 6], [0, 1, 5, 9]], # L
        [[1, 2, 5, 6]], # O
        [[4, 5, 1, 2], [1, 5, 6, 10], [8, 9, 5, 6], [0, 4, 5, 9]], # S
        [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [1, 4, 5, 9]], # T
        
        [[0, 1, 5, 6], [2, 6, 5, 9], [4, 5, 9, 10], [1, 5, 4, 8]] # Z  
        
    ]
    
    # Used for rotational nudges
    # Used for I, J, T, S, Z blocks (some T blocks won't be achieved)
    # O blocks cannot rotate
    kicks_main = {
        (0,1): [(-1,0), (-1,1), (0,-2), (-1,-2)],
        (1,0): [(1,0), (1,-1), (0,2), (1,2)],
        (1,2): [(1,0), (1,-1), (0,2), (1,2)],
        (2,1): [(-1,0), (-1,1), (0,-2), (-1,-2)], 
        (2,3): [(1,0), (1,1), (0,-2), (1,-2)],
        (3,2): [(-1,0), (-1,-1), (0,2), (-1,2)],
        (3,0): [(-1,0), (-1,-1), (0,2), (-1,2)],
        (0,3): [(1,0), (1,1), (0,-2), (1,-2)]
    }

    # Different set of nudges for I blocks
    kicks_I  = {
        (0,1): [(-2,0), (1,0), (-2,-1), (1,2)],
        (1,0): [(2,0), (-1,0), (2,1), (-1,-2)],
        (1,2): [(-1,0), (2, 0), (-1,2), (2,-1)],
        (2,1): [(1,0), (-2,0), (1,-2), (-2, 1)], 
        (2,3): [(2,0), (-1,0), (2,1), (-1,-2)],
        (3,2): [(-2,0), (1,0), (-2,-1), (1,2)],
        (3,0): [(1,0), (-2,0), (1,-2), (-2,1)],
        (0,3): [(-1,0), (2,0), (-1,2), (2,-1)]
    }
        

    def __init__(self, x, y, mode = None):
        self.x = x
        self.y = y
        if not (mode in range(1,len(self.figures)+1)):
            self.type = random.randint(1, len(self.figures)-1) # 0 indexing! self.figures has an empty spot, randint is inclusive
        else:
            self.type = mode
        self.name = self.names[self.type]
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self, direction):
        self.rotation = (self.rotation + direction) % len(self.figures[self.type])