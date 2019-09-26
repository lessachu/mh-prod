from copy import deepcopy, copy
import math

VERTICAL = 'vertical'
HORIZONTAL = 'horizontal'
ON_END = 'on_end'

# just for the whale, which of the six sides is facing up
TAIL_UP = 'tail_up'
FACE_UP = 'face_up'
BELLY_UP = 'belly_up'
BACK_UP = 'back_up'
RIGHT_SIDE_UP = 'right_side_up'
LEFT_SIDE_UP = 'left_side_up'
NORTH = 'north'
SOUTH = 'south'
EAST = 'east'
WEST = 'west'
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
SIZE = {'x': 6, 'y': 6}

# this all assumes a north face

whale_moves = {
    FACE_UP : { UP : (BACK_UP, NORTH), RIGHT : (LEFT_SIDE_UP, EAST), DOWN : (BELLY_UP, SOUTH), LEFT: (RIGHT_SIDE_UP, WEST) },
    BACK_UP : { UP : (TAIL_UP, NORTH), RIGHT : (LEFT_SIDE_UP, NORTH), DOWN : (FACE_UP, NORTH), LEFT: (RIGHT_SIDE_UP, NORTH) },
    TAIL_UP : { UP : (BELLY_UP, SOUTH), RIGHT : (LEFT_SIDE_UP, WEST), DOWN : (BACK_UP, NORTH), LEFT: (RIGHT_SIDE_UP, EAST) },
    BELLY_UP : { UP : (TAIL_UP, SOUTH), RIGHT : (RIGHT_SIDE_UP, NORTH), DOWN : (FACE_UP, SOUTH), LEFT : (LEFT_SIDE_UP, NORTH) },
    RIGHT_SIDE_UP : { UP : (TAIL_UP, WEST), RIGHT : (BACK_UP, NORTH), DOWN : (FACE_UP, EAST), LEFT : (BELLY_UP, NORTH) },
    LEFT_SIDE_UP : { UP : (TAIL_UP, EAST), RIGHT : (BELLY_UP, NORTH), DOWN : (FACE_UP, WEST), LEFT : (BACK_UP, NORTH) } 
}

face_to_move = {
    NORTH : { UP : UP, LEFT : LEFT, DOWN : DOWN, RIGHT : RIGHT },
    EAST : { UP : LEFT, LEFT : DOWN, DOWN : RIGHT, RIGHT : UP },
    SOUTH : { UP : DOWN, LEFT : RIGHT, DOWN : UP, RIGHT : LEFT },
    WEST : { UP : RIGHT, LEFT : UP, DOWN : LEFT, RIGHT : DOWN }
}

delta_from_north = {
    NORTH : 0, WEST : 1, SOUTH : 2, EAST : 3
}

DIRECTIONS = [ NORTH, WEST, SOUTH, EAST]

class WrongInputException(Exception):
    pass


class CanNonSolveException(Exception):
    pass


class Car(object):

    def __init__(
            self, orientation, character, start, stop, is_red_car=None):
        self.orientation = orientation  # VERTICAL or HORIZONTAL or ON_END
        self.whale_orientation = ""
        self.whale_face = ""   # north, south, east or west
        self.character = character
        self.start = start  # {'x': x, 'y': y}
        self.stop = stop  # {'x': x, 'y': y}
        self.is_red_car = is_red_car  # red car we need to free out

    def __deepcopy__(self, memo):
        """
        Overwride default deepcopy to simplify it
        and cover only our case in order to improve performance
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def get_points(self):
        """
        Returns set of points that car uses on board
        """
        points = []
        car = self
        x_start, y_start = car.start['x'], car.start['y']
        x_stop, y_stop = car.stop['x'], car.stop['y']
        character = car.character
        if self.orientation == VERTICAL:
            for y in range(y_start, y_stop + 1):
                points.append({'x': x_start, 'y': y, 'character': character})
        if self.orientation == HORIZONTAL:
            for x in range(x_start, x_stop + 1):
                points.append({'x': x, 'y': y_start, 'character': character})
        if self.orientation == ON_END:
            points.append({'x':x_start, 'y':y_start, 'character': character})

        return points

    def red_car_move(self, direction):
        orientation_dict = whale_moves[self.whale_orientation];
        # need to translate direction based on where the whale is currently facing
        new_direction = face_to_move[self.whale_face][direction]

  #      print "whale is " + self.whale_orientation + " facing " + self.whale_face + " moving " + new_direction

        new_orientation = orientation_dict[new_direction]
        self.whale_orientation = new_orientation[0]

        # similarly, modify face information based on the move
 #       print "Face (from north) is " + new_orientation[1]

        new_face = DIRECTIONS[(DIRECTIONS.index(self.whale_face) + delta_from_north[new_orientation[1]]) % 4]

  #      print "After move, whale is " +  self.whale_orientation + " facing " + new_face

        self.whale_face = new_face


        if direction == DOWN:
            if self.orientation == VERTICAL:
                self.start['y'] += 2
                self.stop['y'] += 1
                self.orientation = ON_END
            elif self.orientation == HORIZONTAL:
                self.start['y'] += 1
                self.stop['y'] += 1
            else:
                self.start['y'] += 1
                self.stop['y'] += 2
                self.orientation = VERTICAL

        if direction == UP:
            if self.orientation == VERTICAL:
                self.start['y'] -= 1
                self.stop['y'] -= 2
                self.orientation = ON_END
            elif self.orientation == HORIZONTAL:
                self.start['y'] -= 1
                self.stop['y'] -= 1
            else:
                self.start['y'] -= 2
                self.stop['y'] -= 1
                self.orientation = VERTICAL

        if direction == LEFT:
            if self.orientation == VERTICAL:
                self.start['x'] -= 1
                self.stop['x'] -= 1
            elif self.orientation == HORIZONTAL:
                self.start['x'] -= 1
                self.stop['x'] -= 2
                self.orientation = ON_END
            else:
                self.start['x'] -= 2
                self.stop['x'] -= 1
                self.orientation = HORIZONTAL

        if direction == RIGHT:
            if self.orientation == VERTICAL:
                self.start['x'] += 1
                self.stop['x'] += 1
            elif self.orientation == HORIZONTAL:
                self.start['x'] += 2
                self.stop['x'] += 1
                self.orientation = ON_END
            else:
                self.start['x'] += 1
                self.stop['x'] += 2
                self.orientation = HORIZONTAL



    def red_car_can_move(self, direction, length, matrix, walls, endx, endy):
        if length > 1:
            return False

        wall_jump = False

        origin_points = self.get_points()

        red_car = deepcopy(self)
        red_car.move(direction, 1)

        destination_points = red_car.get_points()


        # whale can move off the board through the exit gate
        if direction == UP and len(origin_points) == 1:
#           print "whale is trying to move UP on the board from (" + str(origin_points[0]['x']) + "," + str(origin_points[0]['y']) + ")"

            if origin_points[0]['x'] == endx and origin_points[0]['y'] == endy+1:
#               print "whale is standing at (" + str(endx) + ',' + str(endy+1) + ")"

                if ((destination_points[0]['x'] == endx and destination_points[0]['y'] == endy and 
                    destination_points[1]['x'] == endx and destination_points[1]['y'] == endy-1) or 
                    (destination_points[1]['x'] == endx and destination_points[1]['y'] == endy and 
                    destination_points[0]['x'] == endx and destination_points[0]['y'] == endy-1)):

#                   print "trying to go off the board through the exit"

                    #check that nothing is in the way
                    character = matrix[endy][endx]
                    if character != '.' and character != self.character:
#                       print "Way is blocked by another car"
                        return False

                    #handle the wall case
                    if (origin_points[0]['x'], origin_points[0]['y']) in walls:
#                       print "our square has a wall"
                        wall_squares = walls[(origin_points[0]['x'], origin_points[0]['y'])]
                        if ((destination_points[0]['x'], destination_points[0]['y']) in wall_squares) or ((destination_points[1]['x'], destination_points[1]['y']) in wall_squares):

                            # you can only jump over the wall if you are face_up
#                           print "the whale jumped over the wall!"
                            if (self.whale_orientation == FACE_UP and face_to_move[self.whale_face][direction] == UP):
#                               print "Valid move"
                                return True
                            else:
#                               print "Invalid move"
                                return False
                    return True

        for point in destination_points:

      #      print "verify point "  + str(point) + " for " + direction + " move"

            if point['y'] < 0 or point['x'] < 0:
                return False

            if point['y'] > len(matrix) - 1: 
                return False
            if point['x'] > len(matrix) -1:
                return False

            try:
                character = matrix[point['y']][point['x']]
                if character != '.' and character != self.character:
                    return False

                if (point['x'],point['y']) in walls:
                    # print "trying to move into a wall square (" + str(point['x']) + "," + str(point['y']) + ")"

                    # print "whale is " + self.whale_orientation + " " + self.whale_face + " trying to move " + direction
                    # print "face_to_move is " + face_to_move[self.whale_face][direction] 

                    # two special cases for wall squares:
                    # whale can leap over a wall, so it's valid to be in a wall square, if you're facing
                    # the right away

                    wall_squares = walls[(point['x'],point['y'])]
                    for start_point in origin_points:
                        if (start_point['x'],start_point['y']) in wall_squares:

#                           print "whale started in an origin point for the wall"
                            if self.whale_orientation != FACE_UP or face_to_move[self.whale_face][direction] != UP:
                                return False;
                            else:
                                wall_jumped = True

                    # but it's not valid if the move lands you on a wall
                    for point2 in destination_points:
                        if (point2['x'],point2['y']) in wall_squares:
                   #         print "whale is trying to jump onto a wall"
                            return False


            except IndexError:
                return False

        del red_car

 #       print "Red Car move is valid"

        return True

    def can_move(self, direction, length, matrix, walls, endx, endy):
        """
        Check if we can move car to `direction` and `length`
        """

#       print "Trying move " + direction + " for " + self.character + " from " + str(self.start['x']) + "," + str(self.start['y'])

        if self.is_red_car:
            return self.red_car_can_move(direction, length, matrix, walls, endx, endy)

        if self.orientation == HORIZONTAL:
            if direction in [UP, DOWN]:
                return False

        if self.orientation == VERTICAL:
            if direction in [LEFT, RIGHT]:
                return False

        # check if there are some other cars on the way
        # or board ending
        car_start_points = self.get_points()

        car = deepcopy(self)
        car.move(direction, length)
        for point in car.get_points():
            if point['y'] < 0 or point['x'] < 0:
                return False

            if (point['x'],point['y']) in walls:
            #    print "trying to move into a wall square (" + str(point['x']) + "," + str(point['y']) + ")"
                wall_squares = walls[(point['x'],point['y'])]
                for start_point in car_start_points:
                    if (start_point['x'],start_point['y']) in wall_squares:
                        return False;
            try:
                character = matrix[point['y']][point['x']]
                if character != '.' and character != self.character:
                    return False
            except IndexError:
                return False
        del car

#      print "Move is valid"
        return True

    def move(self, direction, length):

        if self.is_red_car:
            self.red_car_move(direction)
            return

        if direction == UP:
            self.start['y'] -= length
            self.stop['y'] -= length

        if direction == DOWN:
            self.start['y'] += length
            self.stop['y'] += length

        if direction == LEFT:
            self.start['x'] -= length
            self.stop['x'] -= length

        if direction == RIGHT:
            self.start['x'] += length
            self.stop['x'] += length

    def __repr__(self):
        return "{} ({}->{},{},{})".format(self.character, self.start, self.stop, self.whale_orientation, self.whale_face)


class Solver(object):

    def __init__(self, size={'x': 6, 'y': 6}):
        super(Solver, self).__init__()
        self.size = size
        self.cars = []
        self.steps = []
        self.endX = 0
        self.endY = 0
        self.wall_matrix = []
        self.wall_edge_dict = {}

    def on_board(self, x, y):
        if x < 0 or x >= self.size['x']:
            return False;
        return not (y < 0 or y >= self.size['y'])

    def single_car(self, matrix, character, x, y):
        top = self.on_board(x, y-1) and matrix[x][y-1] == character
        left = self.on_board(x-1, y) and matrix[x-1][y] == character
        bottom = self.on_board(x, y+1) and matrix[x][y+1] == character
        right = self.on_board(x+1, y) and matrix[x+1][y] == character

        return not (top or left or bottom or right)

    def generate_cars_vertical(self, matrix, whale_start_orientation, whale_start_face):
        """
        Go through line by line and generate cars
        """
        data = []
        for x in range(self.size['x']):
            car_data = {'character': None}
            for y in range(self.size['y']):
                item = matrix[y][x]
  #              print "looking in (" + str(x) + "," + str(y) + ")" + str(car_data) + " " + str(item)

                if item != car_data['character'] or y == self.size['y'] - 1:
                    # We have start/stop point here
                    if car_data['character']:
                        car_data['stop'] = {'x': x, 'y': y - 1}
                        if y == self.size['y'] - 1:
                            if item == car_data['character']:
                                car_data['stop'] = {'x': x, 'y': y}

  #                      print str(car_data)

                        if car_data['stop']['y'] - car_data['start']['y'] > 0:
                            if car_data['character'] != '.':
                                car_obj = Car(
                                    VERTICAL,
                                    car_data['character'],
                                    car_data['start'],
                                    car_data['stop'],
                                    is_red_car=(car_data['character'] == 'r')
                                )

                                if car_obj.is_red_car:
                                    car_obj.whale_orientation = whale_start_orientation
                                    car_obj.whale_face = whale_start_face


                                print("red car is vertical at " + str(car_data['start']['x']) + "," +
                                    str(car_data['stop']['y']))

                                self.cars.append(car_obj)
                    if item == 'r':
                        print "item is r"
                        if self.single_car(matrix, item, y, x):
                            car_obj = Car(
                                ON_END,
                                item,
                                { 'x' : x, 'y' : y},
                                { 'x' : x, 'y' : y},
                                is_red_car= True
                            )

                            print "Found whale on end at (" + str(x) + "," + str(y) +")"
                            car_obj.whale_orientation = whale_start_orientation
                            car_obj.whale_face = whale_start_face

                            self.cars.append(car_obj)

                    car_data = {
                        'start': {'x': x, 'y': y},
                        'character': item,
                    }

    def generate_cars_horizontal(self, matrix, whale_start_orientation, whale_start_face):
        """
        Go through line by line and generate cars
        """
        data = []
        for y, line in enumerate(matrix):
            car_data = {'character': None}
            for x, item in enumerate(line):
                if item != car_data['character'] or x == self.size['x'] - 1:
                    # We have start/stop point here
                    if car_data['character']:
                        car_data['stop'] = {'x': x - 1, 'y': y}
                        if x == self.size['x'] - 1:
                            if item == car_data['character']:
                                car_data['stop'] = {'x': x, 'y': y}

                        if car_data['stop']['x'] - car_data['start']['x'] > 0:
                            if car_data['character'] != '.':
                                car_obj = Car(
                                    HORIZONTAL,
                                    car_data['character'],
                                    car_data['start'],
                                    car_data['stop'],
                                    is_red_car=(car_data['character'] == 'r')
                                )
                                if car_obj.is_red_car:
                                    print "Found whale at (" + str(car_data['start']['x']) + "," + str(car_data['start']['y']) +") and (" + str(car_data['stop']['x']) + "," + str(car_data['stop']['y']) + ")"
                                    car_obj.whale_orientation = whale_start_orientation
                                    car_obj.whale_face = whale_start_face


                                self.cars.append(car_obj)

                    car_data = {
                        'start': {'x': x, 'y': y},
                        'character': item,
                    }
        print self.cars

    def str_to_matrix(self, init_data):
        """
        Covert text into 2D array that will be processed further
        Also generate the matrix of wall data
        """
        matrix = []
        walls = []
        for line in init_data.split("\n"):
            line = line.replace(' ', '').replace('\r', '')
            if not line:
                continue

            matrix_line = []
            wall_line = []
            for item in line:
                if item != '|' and item != '-' and item != 'x':
                    matrix_line.append(item)
                else:
                    wall_line.append(item)
            if matrix_line:
                matrix.append(matrix_line)
            if wall_line:
                walls.append(wall_line)

# too annoying to keep changing this variable
        self.size['y'] = len(matrix)
        self.size['x'] = len(matrix[0])
        print "grid size is " + str(len(matrix)) + " x " + str(len(matrix[0]))

  #      if len(matrix) != self.size['y']:
#         raise WrongInputException("Incorrect board size(y) given")
#        for line in matrix:
#            if len(line) != self.size['x']:
#                raise WrongInputException("Incorrect board size(x) given")

        return matrix, walls

    def add_wall_edge(self, walls, x1, y1, x2, y2):
        if (x1,y1) in walls:
            wall_ends = walls[(x1,y1)]
        else:
            wall_ends = []  
        wall_ends.append((x2,y2))
        walls[(x1,y1)] = wall_ends
                  
        if (x2,y2) in walls:
            wall_ends = walls[(x2,y2)]
        else:
            wall_ends = []
        wall_ends.append((x1,y1))
        walls[(x2,y2)] = wall_ends
        return walls


    def generate_wall_dict(self, walls_matrix):

  #      print "wall matrix size is: " + str(len(walls_matrix)) + " x " + str(len(walls_matrix[0]))
        walls = {}

        for y in range(len(walls_matrix)):
            for x in range(len(walls_matrix[y])):
      #          print "checking for walls at " + str(x) + "," + str(y)
                if walls_matrix[y][x] == 'x':
                    print "found wall at " + str(x) + "," + str(y)
                    if y % 2 == 0:
     #                   print "this is a horizontal wall"
                        # horizontal walls are between the cur square the square to the right
                        walls = self.add_wall_edge(walls, x, y/2, x+1, y/2)
                        print walls

                    else:
        #                print "this is a vertical wall"
                        walls = self.add_wall_edge(walls, x, y/2, x, y/2 + 1)
                        print walls

        return walls

    def load_data(self, board_data, endx, endy, whale_start_orientation, whale_start_face):
        """
        We assume that there is no case
        when car be readed as vertical and horizontal at the same time
        Also we assume that there are no cars on the way of red car
        that can't be moved to side
        """
        results = self.str_to_matrix(board_data)
        matrix = results[0]
        self.wall_matrix = results[1]
        self.walls = self.generate_wall_dict(self.wall_matrix)
        self.generate_cars_horizontal(matrix, whale_start_orientation, whale_start_face)
        self.generate_cars_vertical(matrix, whale_start_orientation, whale_start_face)
        self.check_data(self.cars)
        self.endx = endx
        self.endy = endy

    def check_data(self, cars):
        """
        - check if red car on board
        - check if we have only cars in size > 1
        """
        if not filter(lambda x: x.is_red_car, cars):
            raise WrongInputException("No red car found")

    #    result = tuple(filter(lambda car: len(list(car.get_points())) < 2, cars))
    #    if result:
    #        raise WrongInputException("Car should take at least two cells")

        return

    def get_all_states(self, cars):
        """
        It takes current cars state and generates all possible next states
        within one move of one car
        """
        states = []
        for car in cars:
            for direction in [UP, DOWN, LEFT, RIGHT]:
                if car.can_move(direction, 1, self.cars_to_matrix(cars), self.walls, self.endx, self.endy):
                    # TODO: in case of performance improvemens see here first
                    new_cars = deepcopy(cars)
                    new_car = tuple(filter(
                        lambda x: x.character == car.character, new_cars))[0]
                    new_car.move(direction, 1)
                    states.append([[[car.character, direction]], new_cars])
        return states

    def solve(self):
        '''
        Take initial board and get all possible next boards,
        for each board take all next boards,
        iterate untill solved
        '''
        Q = []
        cars = self.cars
        visited = set()
        Q.append([[], cars])
        best_moves = []
        min_moves = 999999
        is_doubled = False

        while len(Q) != 0:
  #          print "Q : " + str(Q)
            moves, cars = Q.pop(0)

 #           print "checking moves, cars:" + str(moves) + " " + str(cars)

            if self.is_solved(cars):
                if len(moves) < min_moves:
                    is_doubled = False
                    min_moves = len(moves)
                    best_moves = deepcopy(moves)
                elif len(moves) == min_moves:
                    print(solver.format_steps(solver.cars, moves))
                    is_doubled = True

            for new_moves, new_cars in self.get_all_states(cars):
  #              print "Considering:\n" + str(new_cars)
                if hash(str(new_cars)) not in visited:
   #                 print "Adding to Q: " + str( moves + new_moves) + " " + str(new_cars)
                    Q.append([moves + new_moves, new_cars])
                    visited.add(hash(str(new_cars)))

        if len(best_moves) == 0:
            raise CanNonSolveException('Can not solve')
        elif is_doubled:
            print "Multiple best paths found"
 #           raise CanNonSolveException('multiple best paths found')
        else:
            return best_moves

    def is_solved(self, cars):
        """
        Moment when red car is on the right side
        """
        red_car = tuple(filter(lambda x: x.is_red_car, cars))[0]

  #      print "Solved?  whale is at (" + str(red_car.stop['x']) + "," + str(red_car.stop['y']) + "): " + red_car.orientation + " " + red_car.whale_orientation + " " + red_car.whale_face
 
  #      print self.format_data(cars)

       # hacking some solution conditions

        if red_car.whale_orientation in [RIGHT_SIDE_UP, LEFT_SIDE_UP, BELLY_UP, BACK_UP] and red_car.orientation == VERTICAL:
            for point in red_car.get_points():
                if point['x'] == self.endx and point['y'] == self.endy:
                    return True
        else:
            if red_car.stop['x'] == self.endx and red_car.stop['y'] == self.endy:
                return True
        return False

    def format_steps(self, cars, moves):
        num_whale_moves = 0
        output = ''
        output += '\n\nSOLUTION\n'
        output += "; ".join(["{} {}".format(move[0], move[1]) for move in moves])
        cars = deepcopy(cars)
        for move in moves:
            car = tuple(filter(lambda x: x.character == move[0], cars))[0]
            output += '\nMOVE {} {}\n'.format(move[0], move[1])
            if car.character == 'r':
                num_whale_moves = num_whale_moves + 1
                output += "\nWhale is " + car.whale_orientation + " facing " + car.whale_face + "\n"

            car.move(move[1], 1)
            output += self.format_data(cars)
        output += '\nEND of SOLUTION\n\n' + 'Total Moves: ' + str(len(moves)) + ' Whale Moves: ' + str(num_whale_moves) 
        return output

    def cars_to_matrix(self, cars):
        data = []
        for y in range(self.size['y']):
            line = []
            for x in range(self.size['x']):
                line.append('.')
            data.append(line)

        for car in cars:
            for point in car.get_points():
                if self.on_board(point['x'],point['y']):
                    data[point['y']][point['x']] = point['character']

        return data

    def format_data(self, cars):
        matrix = self.cars_to_matrix(cars)
        wall_line = []

        # add in sea wall information
        for y in reversed(range(self.size['y'])):
            wall_line = []
            for x in reversed(range(self.size['x'] - 1)):
                matrix[y].insert(x+1," ")
                wall_line.append('  ')
            wall_line.append(' ')
            if y < self.size['y'] - 1:
                matrix.insert(y+1, wall_line)

        for y in range(len(self.wall_matrix)):
            for x in range(len(self.wall_matrix[y])):
                if self.wall_matrix[y][x] == 'x':
                    if y % 2 == 0:
                        matrix[y][(x*2) + 1] = '|'
                    else:
                        matrix[y][x] = '- '

        output = ''
        for line in matrix:
            output += "".join(line) + '\n' 
        return output


if __name__ == '__main__':

# puzzle 3
    # wall_data = '''
    #     B|B|B|.|.|C
    #     - - x - - -
    #     A|A|.|.|D|C
    #     - - - - - -
    #     .|.|.|.|D|C
    #     - x - x - - 
    #     E|r|F|F|F|F
    #     - - - - - -
    #     Ex.|.|.x.|.
    #     - - - - - x
    #     E|.|.|.|.|.
    #    '''


    # endx = 3
    # endy = 0

    # whale_start_orientation = FACE_UP
    # whale_start_face = NORTH

    wall_data = '''
        .|.|.|A|.|.
        - x - - - -
        .|.|.|A|.|.                                                                                                                                                                     .
        x x x - x -
        .|.|.|A|.x.
        - - - - - x 
        C|C|.|.|.|.
        - x - - - -
        .|.|.|B|B|B
        - - - x - -
        .|.|.|.|r|r
       '''


    endx = 2
    endy = 0

    print "end is at " + str(endx) + "," + str(endy)

    whale_start_orientation = BACK_UP
    whale_start_face = WEST

    # wall_data = '''
    #    .|.|.
    #    - x -
    #    .|.|.
    #    - - x 
    #    .|.xr
    # '''


    # endx = 1
    # endy = 0

    # wall_data = '''
    #     .x.x.|.|.|.
    #     - x - - - -
    #     .|.|.|.|.|.                                                                                                                                                                     .
    #     - - - - - -
    #     .|.|.xA|A|.
    #     - - - - - - 
    #     .|.|.|.|r|.
    #     - - - x - -
    #     .|.|.|.|r|.
    #     - - - - - -
    #     .|.|.|.x.|.
    #    '''


    # endx = 1
    # endy = 0



    # whale_start_orientation = LEFT_SIDE_UP
    # whale_start_face = NORTH

    print "end is at " + str(endx) + "," + str(endy)

    solver = Solver()
    solver.load_data(wall_data, endx, endy, whale_start_orientation, whale_start_face)
    print('Loaded data')
    solver.format_data(solver.cars)
    print('Looking for solution.. (may take several seconds)')
    moves = solver.solve()
    print(solver.format_steps(solver.cars, moves))
