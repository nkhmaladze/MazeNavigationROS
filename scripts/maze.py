import numpy as np
import heapq
'''A simple module to store and query 2D mazes with grid-aligned walls.'''

DIR_LEFT = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_UP = 3

DIR_RC_DELTAS = [
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0)
]

DIR_XY_DELTAS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

DIR_STRINGS = [
    'left',
    'right',
    'down',
    'up'
]

DIR_STRING_LOOKUP = dict(zip(DIR_STRINGS, range(4)))

def split_command(str_or_list):
    '''Splits a command string into five integer tokens. For instance, the
string

  0 0 up 3 5

would return the tuple 0, 0, 3, 3, 5 (because DIR_UP=3).

    '''

    if isinstance(str_or_list, str):
        x0, y0, dir0, x1, y1 = str_or_list.split()
    else:
        x0, y0, dir0, x1, y1 = str_or_list
        
    return int(x0), int(y0), DIR_STRING_LOOKUP[dir0.lower()], int(x1), int(y1)

def add_tuple(x,y):
    return tuple(np.array(x) + np.array(y))

def path_actions(path, initial_orientation):
    '''Gets the actions to follow the path given the initial
    orientation. The path argument should be a list of (x, y)
    locations, each one horizontally or vertically adjacent to its
    predecessor/successor, and initial_orientation should be one of
    DIR_UP, DIR_DOWN, DIR_LEFT, or DIR_RIGHT. The path should be a
    list of the strings 'forward', 'turnleft', or 'turnright' in the
    correct order.

    ''' 
    actions = [ ]
    current_orientation = initial_orientation
    for i in range(len(path)-1):
        if current_orientation == DIR_RIGHT:
            if path[i+1] == add_tuple(path[i],(1,0)) :
                actions.append('forward')
                
            elif path[i+1] == add_tuple(path[i],(0,-1)):
                actions.append('turnright')
                actions.append('forward')
                current_orientation = DIR_DOWN

            elif path[i+1] == add_tuple(path[i],(0,1)):
                actions.append('turnleft')
                actions.append('forward')
                current_orientation = DIR_UP

        elif current_orientation == DIR_LEFT:
            if path[i+1] == add_tuple(path[i],(-1,0)) :
                actions.append('forward')

            elif path[i+1] == add_tuple(path[i],(0,-1)):
                actions.append('turnleft')
                actions.append('forward')
                current_orientation = DIR_DOWN

            elif path[i+1] == add_tuple(path[i],(0,1)):
                actions.append('turnright')
                actions.append('forward')
                current_orientation = DIR_UP
        
        elif current_orientation == DIR_UP:
            if path[i+1] == add_tuple(path[i],(0,1)) :
                actions.append('forward')

            elif path[i+1] == add_tuple(path[i],(-1,0)):
                actions.append('turnleft')
                actions.append('forward')
                current_orientation = DIR_LEFT

            elif path[i+1] == add_tuple(path[i],(1,0)):
                actions.append('turnright')
                actions.append('forward')
                current_orientation = DIR_RIGHT
        
        elif current_orientation == DIR_DOWN:
            if path[i+1] == add_tuple(path[i],(0,-1)) :
                actions.append('forward')

            elif path[i+1] == add_tuple(path[i],(-1,0)):
                actions.append('turnright')
                actions.append('forward')
                current_orientation = DIR_LEFT

            elif path[i+1] == add_tuple(path[i],(1,0)):
                actions.append('turnleft')
                actions.append('forward')
                current_orientation = DIR_RIGHT
    #print(" These are our output actions haha xd: \n")
    #print(actions)
    return actions

class Maze(object):

    '''Encapsulate a maze and support querying/setting positions of walls.
When a maze is read in from a file or printed, the bottom-left corner
corresponds to (x, y) = (0, 0). The x coordinate increases to the
right and the y coordinate increases going up.

    '''

    def __init__(self, size_or_filename=None):

        '''Initializer. You can pass in a size in as a (width, height) tuple
to create an empty maze of that size, or a filename of a file to
read. Otherwise, creates an empty 0x0 maze. You can call load() or
create() later to finish constructing the maze.

        '''

        self.data = None

        if isinstance(size_or_filename, tuple):
            self.create(*size_or_filename)
        elif isinstance(size_or_filename, str):
            self.load(size_or_filename)
        else:
            self.clear()

    def _rc_valid(self, r, c):
        '''Internal helper function to check if row/col valid.'''
        return (r >= 0 and c >= 0 and
                r < self.data.shape[0] and
                c < self.data.shape[1])

    def _xy_to_rc(self, x, y):
        '''Internal helper function to convert x/y to row/col.'''
        r = self.data.shape[0]-1-2*y
        c = 2*x
        return r, c

    def width(self):
        '''Returns the width of this maze.'''
        return (self.data.shape[1]+1)/2

    def height(self):
        '''Returns the height of this maze.'''
        return (self.data.shape[0]+1)/2

    def clear(self):
        '''Resets the maze to an empty 0x0 maze.'''
        self.data = np.empty((0, 0), dtype=bool)

    def create(self, width, height):
        '''Create a blank maze of the given size. No walls will be added.'''
        if width <= 0 or height <= 0:
            raise RuntimeError('cannot create empty maze; use clear() instead.')
        self.data = np.zeros((2*height-1, 2*width-1), dtype=bool)

    def add_wall(self, x, y, dir_index):
        '''Add a wall to the given (x, y) location on the border with the
given direction. Invalid locations/directions will fail silently.'''
        self.set_wall(x, y, dir_index, True)

    def remove_wall(self, x, y, dir_index):
        '''Remove a wall from the given (x y) location on the border with the
given direction. Invalid locations/directions will fail silently.'''
        self.set_wall(x, y, dir_index, False)

    def set_wall(self, x, y, dir_index, wval):

        '''Add or remove a wall to the given (x, y) location on the border
with the given direction. Invalid locations/directions will fail
silently.

        '''

        r0, c0 = self._xy_to_rc(x, y)

        delta_r, delta_c = DIR_RC_DELTAS[dir_index]

        r1, c1 = r0+delta_r, c0+delta_c

        if self._rc_valid(r0, c0) and self._rc_valid(r1, c1):
            self.data[r1, c1] = bool(wval)

    def can_move(self, x, y, dir_index):

        '''Returns true if it is possible to move in the given direction from
the cell (x, y).'''

        r, c = self._xy_to_rc(x, y)

        delta_r, delta_c = DIR_RC_DELTAS[dir_index]

        for _ in range(3):
            if not self._rc_valid(r, c) or self.data[r, c]:
                return False
            r += delta_r
            c += delta_c

        return True

    def reachable_neighbors(self, x, y):

        '''Return a list of all (xn, yn) positions for neighbors of the cell
at (x, y).'''

        r, c = self._xy_to_rc(x, y)

        rval = []

        if not self._rc_valid(r, c):
            return rval

        for dir_index in range(4):
            if self.can_move(x, y, dir_index):
                dx, dy = DIR_XY_DELTAS[dir_index]
                rval.append((x+dx, y+dy))

        return rval

    def pretty(self):

        '''Return a string representing a "pretty-print" of the maze with the
characters +|- as well as space and newline.'''

        outstr = []

        nrows, ncols = self.data.shape

        outstr.append(('+-' * ((ncols+1)/2)) + '+\n')

        for i in range(nrows):
            outstr.append('|')
            for j in range(ncols):
                if not self.data[i, j]:
                    outstr.append(' ')
                elif i % 2 == 0 and j % 2 == 0:
                    outstr.append('#')
                elif i % 2 == 0 and j % 2 == 1:
                    outstr.append('|')
                elif i % 2 == 1 and j % 2 == 0:
                    outstr.append('-')
                else:
                    outstr.append('+')
            outstr.append('|\n')

        outstr.append(('+-' * ((ncols+1)/2)) + '+\n')

        return ''.join(outstr)

    def load(self, filename):

        '''Load a maze from a file with the given name.'''

        with open(filename, 'r') as f:
            array_of_strings = [line.strip() for line in f]
        self.load_from_array(array_of_strings, filename)

    def load_from_string(self, string):
        '''Load a maze from a Python string in memory.'''

        return self.load_from_array(self, string.split('\n'))

    def load_from_array(self, array_of_strings, filename='maze input'):

        '''Load a maze from an array of strings. You can provide an optional
filename which will appear in any error messages produced by this
function.'''

        nrows = len(array_of_strings)
        ncols = len(array_of_strings[0])

        if nrows < 3 or ncols < 3:
            raise RuntimeError('{}: error: empty maze!'.format(filename))

        self.data = np.zeros((nrows-2, ncols-2))

        for i in range(nrows):

            if len(array_of_strings[i]) != ncols:
                raise RuntimeError('{}:{} wrong length '
                                   '(expected {}, but got {})'.format(
                                       filename, i+1, ncols,
                                       len(array_of_strings[i])))

            for j in range(ncols):

                cij = array_of_strings[i][j]

                if (i == 0 or i+1 == nrows or
                    j == 0 or j+1 == ncols or
                    (i % 2 == 0 and j % 2 == 0)):

                    if cij.isspace():
                        print '{}:{}:{} warning: expected non-space'.format(
                            filename, i+1, j)

                elif i % 2 == 1 and j % 2 == 1:

                    if not cij.isspace():
                        print '{}:{}:{} warning: expected space'.format(
                            filename, i+1, j)

                if i > 0 and j > 0 and i+1 < nrows and j+1 < ncols:
                    self.data[i-1, j-1] = not cij.isspace()

    def is_solvable(self):

        '''Return true if the maze is solvable -- that is, every other cell
can be reached from position (0, 0).'''

        init_pos = (0, 0)

        queue = [init_pos]
        visited = set(queue)

        while len(queue):
            x, y = queue.pop()
            for n in self.reachable_neighbors(x, y):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)

        for y in range(self.height()):
            for x in range(self.width()):
                if (x, y) not in visited:
                    return False

        return True 

    def solve(self, x0, y0, x1, y1):

        '''Returns a list of (x, y) locations along the shortest path
        from (x0, y0) to (x1, y1), or None if no path exists. If the
        search succeeds, the returned path starts with (x0, y0) and
        ends with (x1, y1).
        '''
        start = ( x0, y0 )
        end = ( x1, y1 )

        pq = [ ( 0, start )]
        distances =  { start : 0 }
        previous_pos = { start : None }

        while pq:
                current_distance, current_pos = heapq.heappop( pq )

                if current_pos == end:
                    path = []
                    while current_pos:
                        path.append( current_pos )
                        current_pos = previous_pos[ current_pos ]
                    return path[::-1]
                
                for neighbor in self.reachable_neighbors( *current_pos ):
                    distance = current_distance + 1 
                    if neighbor not in distances or distance < distances[neighbor]:
                        distances[neighbor] = distance 
                        previous_pos[neighbor] = current_pos
                        heapq.heappush( pq, (   distance, neighbor ) )

        return None

def _check_path(m, x0, y0, x1, y1, path):

    return ok


def _do_tests():


    test_cmds = [
        '0 0 up 4 5',
        '3 2 down 1 2',
        '0 2 left 0 4',
        '1 1 right 1 2'
        ]

    for test_cmd in test_cmds:
        print test_cmd, '->', split_command(test_cmd)

    print

    test_maze_1 = [
        '+-+-+-+-+',
        '| |     |',
        '+ + +-+ +',
        '| | |   |',
        '+ + +-+-+',
        '|       |',
        '+-+-+ +-+',
        '|       |',
        '+ + +-+-+',
        '| |     |',
        '+ +-+-+ +',
        '| |     |',
        '+-+-+-+-+',
    ]
    
    test_maze_2 = [
        '+-+-+-+',
        '| |   |',
        '+ + +-+',
        '|   | |',
        '+-+-+-+'
    ]

    test_mazes = [
        test_maze_1,
        test_maze_2,
    ]
    
    fail = False

    for test_maze in test_mazes:

        print '*'*50
        print

        m = Maze()
        m.load_from_array(test_maze)

        h = m.height()
        w = m.width()

        print 'maze is {} by {}:'.format(w, h)
        print
        print m.pretty()

        print
        print 'maze solvable:', m.is_solvable()
        print
        
        x0 = 0
        y0 = h-1

        x1 = w-1
        y1 = 0

        print 'searching for path from ({}, {}) to ({}, {})...'.format(x0, y0, x1, y1)
        path = m.solve(x0, y0, x1, y1)

        if path is None and m.is_solvable():
            print 'expected solution but none found!!!!'
            fail = True
        elif path is not None and not m.is_solvable():
            print 'got solution to unsolvable maze!!!!'
            fail = True

        if path is None:

            print 'no path found'

        else:

            print 'got path ', path

            print 'starting facing down, path actions are:', path_actions(path, DIR_DOWN)

            if not len(path):
                print 'empty path!'
    

            if path[0] != (x0, y0):
                print 'path does not start with ({}, {})'.format(x0, y0)
                fail = True

            for i, (xa, ya) in enumerate(path[:-1]):

                (xb, yb) = path[i+1]

                n = m.reachable_neighbors(xa, ya)

                if (xb, yb) not in n:
                    print 'pos {} at ({}, {}) not adjacent to successor ({}, {})'.format(
                        i, xa, ya, xb, yb)
                    fail = True

            if path[-1] != (x1, y1):
                print 'path does not end with ({}, {})'.format(x1, y1)
                fail = True


        print
        

        for y in range(h):
            for x in range(w):
                print 'neighbors of {} are {}'.format(
                    (x, y), m.reachable_neighbors(x, y))

        print

        for wval in [True, False]:

            for y in range(h):
                for x in range(w):
                    for d in range(4):
                        m.set_wall(x, y, d, wval)

            print m.pretty()
            print 'maze solvable:', m.is_solvable()
            print

            assert m.is_solvable() == (not wval)

            print 'solving again...'
            path = m.solve(x0, y0, x1, y1)

            if (path is None) != wval:
                fail = True
                if wval:
                    print 'expected no solution but found one'
                else:
                    print 'expected solution but found none'
            elif path is not None:
                print 'got a solution!'
            else:
                print 'got no solution!'

            print

    if fail:
        print 'not all tests passed :('
    else:
        print 'all tests passed :)'

if __name__ == '__main__':

    _do_tests()
