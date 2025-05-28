
import pygame
import figure
Figure = figure.Figure

class Tetris:
    buffer = 4 # how many rows above actual grid to birth tetrominos
    n_queue = 3 # how many next tetrominos to show coming up
    
    # THIS SHOULD BE CLEANED UP IN DISPLAY STUFF, tetris doesn't need to know about this
    # coordinates of board top left on larger screen
    x = 200
    y = 100
    
    height = 20
    width = 10
    
    lines_per_level = 10 # num lines to get to next level
    points_per_line = [100, 300, 500, 800] # multiplied by level
    # number of frames (fps=60) that each level plays at, not level 29+ is nearly impossible for humans so may need to adjust
    level_frames_per_drop = {0:48, 
                             1:43, 
                             2:38, 
                             3:33, 
                             4:28, 
                             5:23, 
                             6:18, 
                             7:13, 
                             8:8, 
                             9:6, 
                             10:5, 11:5, 12: 5,
                             13:4, 14:4, 15:4,
                             16:3, 17:3, 18:3,
                             19:2, 20:2, 21:2, 22:2, 23:2, 24:2, 25:2, 26:2, 27:2, 28:2,
                             29:1}
        
    max_level = 29   
        
    # tetromino starting coordinates with buffer, 4 right, 2 down
    tet_x = 4
    tet_y = 2
    
    
    
    # CLEAN: TETRIS DOESN'T NEED EXTERNAL DISPLAY INFO
    # coordinates of swap piece with respect to top left of SCREEN
    swap_x = 50
    swap_y = 200
    
    # Queue coordinates top piece
    queue_x = 500
    queue_y = 250
    zoom = 25 # size of grid squares    
    
    # BASICALLY RESET FUNCTION, should that be a separate function?
    def __init__(self, player = 0):
        self.player = player # 0 human, 1 computer
        self.full_height = self.height + self.buffer # including buffer region above
        
        self.landed_blocks = 0 # a metric for reward, small score for landing blocks
        self.board = [] # the main part of the state space (the grid of blocks)
        self.score = 0 
        self.lines = 0
        self.level = 0 # used for speed
        self.frames_per_drop = self.level_frames_per_drop[self.level]
        self.state = "start" # "start" and "gameover" are options
        
        # Only allow not showing game if computer is playing for training

        
        self.queue = [] # contains next pieces
        self.has_swapped = False # to allow user to save a tetromino to swap for later, once per drop
        self.swap_piece = None 
        
        
        # NOT USED NOW, call get_reward() instead
        self.total_reward = 0 # Used for training, always displayed
        
        # Set waiting queue and current figure
        self.figure =  Figure(x=self.tet_x,  y=self.tet_y) # piece currently dropping
        for i in range(self.n_queue):
            self.queue.append(Figure(x=self.tet_x,  y=self.tet_y))

        # Set board to be all 0's, including buffer region above
        for i in range(self.full_height): # add rows above game matrix
            new_line = []
            for j in range(self.width):
                new_line.append(0)
                     
            # TESTING KICKS, this example: https://tetris.fandom.com/wiki/SRS:
                # if (i,j) in [(23,0),(23,1), (23,2), (23,3), (23,4), (23,6), (23,7), (23,8), (23,9),
                #              (22,0),(22,1), (22,2), (22,3), (22,6), (22,7), (22,8), (22,9),
                #              (21,0),(21,1), (21,6), (21,7), (21,8), (21,9),
                #              (20,1), (20,2), (20,3), (20,7), (20,8), (20,9),
                #              (19,6), (19,7), (19,8), (19,9),
                #              (18,6), (18,7), (18,5),
                #              (17,4), (17,5)]:
                #     new_line.append(1)
                # else:
                #     new_line.append(0)
        
            self.board.append(new_line)

        
        
        
        
        
        
    # called in init, when blocks freeze/line break, and debug call for chosen tetromino
    def new_figure(self, mode=None):
        # if short on blocks, add them to queue, pop the first one from the list, add another to end
        while len(self.queue) < self.n_queue:
            self.queue.append(Figure(x=self.tet_x,  y=self.tet_y))
        if mode is None:
            self.figure = self.queue.pop(0)
            self.queue.append(Figure(x=self.tet_x,  y=self.tet_y))
        else:
            self.figure = Figure(x=self.tet_x,  y=self.tet_y, mode=mode) # 
        

    def intersects(self):
        intersection = False
        for ind in self.figure.image():
            i = ind//4
            j = ind%4
            # check was > 0, but I want negative numbers to eventually check if block has been broken for gold/silver squares
            if i + self.figure.y >= self.full_height or \
                    j + self.figure.x >= self.width or \
                    j + self.figure.x < 0 or \
                    self.board[i + self.figure.y][j + self.figure.x] != 0: 
                intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        # original code from https://levelup.gitconnected.com/writing-tetris-in-python-2a16bddb5318
        for i in range(self.buffer, self.full_height):
            zeros = 0
            for j in range(self.width):
                if self.board[i][j] == 0:
                    zeros += 1
                    break
            if zeros == 0:
                lines += 1
                self.lines += 1
                if self.lines % 10 == 0:
                    self.level += 1
                    
                    
                    # POTENTIAL ISSUE IF COUNTER HITS RIGHT MODULO AND DROPS BLOCK IMMEDIATELY HERE
                    self.frames_per_drop = self.level_frames_per_drop[min(self.level, self.max_level)]
                    # udpate frames per drop variable next
                # CHECK RANGE BOUNDARIES, seems to stop at second row 
                # at top
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.board[i1][j] = self.board[i1 - 1][j]
       
    
        # TO DO: FIX SCORE FUNCTION
        #  Include Gold/Silver blocks
        if lines > 0:
            self.score += self.points_per_line[lines-1] * (self.level+1) # lines - 1 and self.level + 1 because of 0 indexing
        return lines

        
    # returns the y value of where piece will drop, used for drawing shadow block
    def shadow_height(self): 
        old_y = self.figure.y    
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        new_y = self.figure.y
        self.figure.y = old_y
        return new_y
        
    def go_space(self):
        if self.state == "gameover":
            return 0
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        return self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            return self.freeze()
        else:
            return 0 # lines
    # def get_reward(self):
    #     self.total_reward = self.score + self.landed_blocks #(self.score + self.landed_blocks) / ((self.landed_blocks+1)/4) 
    #     if self.state == 'gameover':
    #         self.total_reward -= 10
    #     return self.total_reward
           # / ((pygame.time.get_ticks()-game.game_start_time)/1000) 

    def freeze(self):
        for ind in self.figure.image():
            i = ind//4
            j = ind%4
            self.board[i + self.figure.y][j + self.figure.x] = self.figure.piece
            if self.board[self.buffer-1][j + self.figure.x] != 0:
                self.state = "gameover"
        self.landed_blocks += 4
        lines = self.break_lines()
        self.has_swapped = False
        
        if self.state == "gameover":
            return lines
        self.new_figure()
        return lines
            

    def swap(self):
        if self.has_swapped:
            return False
        
        self.has_swapped = True
        # save current piece for later
        if not(self.swap_piece):
            self.swap_piece = Figure(x=self.tet_x,  y=self.tet_y, mode= self.figure.piece)
            self.new_figure()
        else:
            temp_piece = Figure(x=self.tet_x,  y=self.tet_y, mode= self.swap_piece.piece)
            self.swap_piece = Figure(x=self.tet_x,  y=self.tet_y, mode= self.figure.piece)
            self.figure = temp_piece
        return True
        

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self, direction):
        # no rotation for the O block
        if self.figure.name == 'O':
            return
        
        old_rotation = self.figure.rotation
        self.figure.rotate(direction)
        
        # BASE CASE, no kick needed
        if not(self.intersects()):
            return
        
        # Failed without nudge
        #   4 states for each block, not magic number
        new_rotation = (old_rotation + direction)%4 
        
        if self.figure.name == 'I':
            nudges = self.figure.kicks_I[(old_rotation, new_rotation)]
        else:
            nudges = self.figure.kicks_main[(old_rotation, new_rotation)]
        
        old_x = self.figure.x
        old_y = self.figure.y
        
        for nudge in nudges:
            self.figure.x += nudge[0]
            self.figure.y += nudge[1]
            if not (self.intersects()):
                return
            else:
                self.figure.x = old_x
                self.figure.y = old_y
        
        # IF no nudges work
        self.figure.rotation = old_rotation
        
    
    
    
    
    
    
    
    def get_heights(self): # returns list of heights of each column
        heights = [0]*self.width
        
        for col in range(self.width):
            for row in range(self.buffer, self.full_height):
                if self.board[row][col] != 0:
                    heights[col] = self.full_height-row
                    break
        return heights
    
            
            
    def get_lines(self):
        lines = 0
        for row in range(self.full_height):
            line_cleared = True
            for col in range(self.width):
                if self.board[row][col] == 0:
                    line_cleared = False
                    break
            if line_cleared:
                lines += 1
        return lines
        
    def get_holes(self, heights): # sum of empty blocks below filled blocks
        holes = 0
        for col, height in enumerate(heights): # one height for each column
            # skip row 0; holes only count beneath existing blocks
            for row in range(1, height):
                if self.board[self.full_height - row][col] == 0:
                    
                    holes += 1
        # if height is 9, need to check rows 24-1(bottom), 
                                        #    24-2(sec bottom)
                                        #... 24-8
    
        return holes
    
    
    def get_bumps(self, heights):
        bumps = 0
        for col in range(1, self.width):
            bumps += abs(heights[col-1] - heights[col])
        return bumps
    
    # literal heights, NOT row values
    def get_total_height(self, heights):
        return sum(heights) 
        
    
    
    
    
    
    
    
    
    
    
    
    
    # SHOULD BE CLEANED, had to put swap piece here so land function could clean up board
    def get_properties(self, swap_piece): # board should already be modified!
        heights = self.get_heights()
        return [self.get_lines(), self.get_holes(heights),
                self.get_bumps(heights), self.get_total_height(heights), swap_piece, self.queue[1].piece]
    
    # mess with action sequences to get resulting boards and return to original state
    # call get_property function to get properties
    def get_landing_scores(self, action_seq): 
        # Return 6 numbers
        # (lines_cleared, holes, bumpiness, total_height, swap, 1st_queue_piece) 
        # IGNORE REST OF THE BOARD
        swap = action_seq[0]
        rot = action_seq[1]
        shift = action_seq[2]
        
        # define the piece
        if swap == 1 and not(self.has_swapped):
            if self.swap_piece:
                piece = Figure(x=self.tet_x + shift,  y=self.tet_y, mode= self.swap_piece.piece, rotation= rot)
            else:
                piece = Figure(x=self.tet_x + shift,  y=self.tet_y, mode= self.queue[0].piece, rotation = rot)
        else: #not swap, use current piece
            # piece = Figure(x=self.tet_x + shift,  y=self.tet_y, mode= self.queue[0].piece, rotation = rot)
            piece = Figure(x=self.tet_x + shift,  y=self.tet_y, mode= self.figure.piece, rotation = rot)

        # drop piece and check that piece is in bounds, otherwise return False
        old_fig = Figure(x=self.figure.x, y = self.figure.y, mode = self.figure.piece, rotation = self.figure.rotation)
        self.figure = Figure(x = piece.x, y = piece.y, mode = piece.piece, rotation = piece.rotation)
        try:
            if self.intersects(): # may attempt pushing piece out of in legal intersection
                self.figure = Figure(x = old_fig.x, y = old_fig.y, mode = old_fig.piece, rotation = old_fig.rotation)
                return False
        except:
            # may attempt push piece for error in intersection function
            self.figure = Figure(x = old_fig.x, y = old_fig.y, mode = old_fig.piece, rotation = old_fig.rotation)
            return False
        
        # move piece down all the way
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        
        # place block on board
        for ind in self.figure.image():
            i = ind//4
            j = ind%4
            self.board[i + self.figure.y][j + self.figure.x] = self.figure.piece
            
            
        # No swap, keep old swap piece
        if self.has_swapped or swap == 0: # no swap
            if self.swap_piece:
                swap_piece = self.swap_piece.piece
            else:
                swap_piece = 0
        else:
            swap_piece = self.figure.piece      
        properties = self.get_properties(swap_piece)
        
        # cleaning up board
        for ind in self.figure.image():
            i = ind//4
            j = ind%4
            self.board[i + self.figure.y][j + self.figure.x] = 0
        # returning figure
        self.figure = Figure(x = old_fig.x, y = old_fig.y, mode = old_fig.piece, rotation = old_fig.rotation)
        
              
            
        return properties 
        
         
            
    
    
    
    def get_next_states(self): # ASSUMING enough space on board and time to make moves!
        if self.state == 'gameover':
            return None
        states = {}
        action_seqs = []
        for rot in [-1,0,1,2]: # possible rotations
            for shift in range(-6,4): # possible shifts
                
                
                if not(self.figure.name == 'O' and rot != 0) and \
                    not(self.figure.name != 'I' and shift==-6) and \
                    not(self.figure.name in ['I', 'S', 'Z'] and rot in [-1,2]):
                        action_seq = (0, rot, shift)
                        state_summary = self.get_landing_scores(action_seq)
                        if state_summary:
                            states[action_seq] = state_summary
                
                if not(self.has_swapped): # add the sequence with swapping
                    temp = self.swap_piece
                    if not(temp):
                        temp = self.queue[0] # a fig
                        if not temp:
                            continue # Don't think this can happen
                    
                    if not(temp.name == 'O' and rot != 0) and \
                    not(temp.name != 'I' and shift==-6) and \
                    not(temp.name in ['I', 'S', 'Z'] and rot in [-1,2]):
                        action_seq = (1, rot, shift)
                        state_summary = self.get_landing_scores(action_seq)
                        if state_summary:
                            states[action_seq] = state_summary
                
        return states