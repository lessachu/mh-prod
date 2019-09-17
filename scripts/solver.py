from copy import deepcopy, copy
import math

VERTICAL = 'vertical'
HORIZONTAL = 'horizontal'
ON_END = 'on_end'
SIZE = {'x': 6, 'y': 6}


class WrongInputException(Exception):
    pass


class CanNonSolveException(Exception):
    pass


class Car(object):

    def __init__(
            self, orientation, character, start, stop, is_red_car=None):
        self.orientation = orientation  # VERTICAL or HORIZONTAL
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
        if direction == 'down':
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

        if direction == 'up':
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

        if direction == 'left':
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

        if direction == 'right':
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

    def red_car_can_move(self, direction, length, matrix):
        if length > 1:
            return False

        red_car = deepcopy(self)
        red_car.move(direction, 1)
        for point in red_car.get_points():
            if point['y'] < 0 or point['x'] < 0:
                return False
            try:
                character = matrix[point['y']][point['x']]
                if character != '.' and character != self.character:
                    return False
            except IndexError:
                return False

        del red_car

        return True

    def can_move(self, direction, length, matrix):
        """
        Check if we can move car to `direction` and `length`
        """
        if self.is_red_car:
            return self.red_car_can_move(direction, length, matrix)

        if self.orientation == HORIZONTAL:
            if direction in ['up', 'down']:
                return False

        if self.orientation == VERTICAL:
            if direction in ['left', 'right']:
                return False

        # check if there are some other cars on the way
        # or board ending
        car = deepcopy(self)
        car.move(direction, length)
        for point in car.get_points():
            if point['y'] < 0 or point['x'] < 0:
                return False
            try:
                character = matrix[point['y']][point['x']]
                if character != '.' and character != self.character:
                    return False
            except IndexError:
                return False
        del car

        return True

    def move(self, direction, length):

        if self.is_red_car:
            self.red_car_move(direction)
            return

        if direction == 'up':
            self.start['y'] -= length
            self.stop['y'] -= length

        if direction == 'down':
            self.start['y'] += length
            self.stop['y'] += length

        if direction == 'left':
            self.start['x'] -= length
            self.stop['x'] -= length

        if direction == 'right':
            self.start['x'] += length
            self.stop['x'] += length

    def __repr__(self):
        return "{} ({}->{})".format(self.character, self.start, self.stop)


class Solver(object):

    def __init__(self, size={'x': 6, 'y': 6}):
        super(Solver, self).__init__()
        self.size = size
        self.cars = []
        self.steps = []
        self.endX = 0
        self.endY = 0
        self.wall_matrix = []

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

    def generate_cars_vertical(self, matrix):
        """
        Go through line by line and generate cars
        """
        data = []
        for x in range(self.size['x']):
            car_data = {'character': None}
            for y in range(self.size['y']):
                item = matrix[y][x]

                if item != car_data['character'] or y == self.size['y'] - 1:
                    # We have start/stop point here
                    if car_data['character']:
                        car_data['stop'] = {'x': x, 'y': y - 1}
                        if y == self.size['y'] - 1:
                            if item == car_data['character']:
                                car_data['stop'] = {'x': x, 'y': y}

                        if car_data['stop']['y'] - car_data['start']['y'] > 0:
                            if car_data['character'] != '.':
                                car_obj = Car(
                                    VERTICAL,
                                    car_data['character'],
                                    car_data['start'],
                                    car_data['stop'],
                                    is_red_car=(car_data['character'] == 'r')
                                )

                                self.cars.append(car_obj)
                        elif car_data['character'] == 'r':

                            if self.single_car(matrix, car_data['character'], car_data['start']['x'], car_data['start']['y']):
                                car_obj = Car(
                                    ON_END,
                                    car_data['character'],
                                    car_data['start'],
                                    car_data['stop'],
                                    is_red_car= True
                                )

                                print("red car is on an end at " + str(car_data['start']['x']) + "," +
                                    str(car_data['stop']['y']))
                                self.cars.append(car_obj)

                    car_data = {
                        'start': {'x': x, 'y': y},
                        'character': item,
                    }

    def generate_cars_horizontal(self, matrix):
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
                                self.cars.append(car_obj)

                    car_data = {
                        'start': {'x': x, 'y': y},
                        'character': item,
                    }

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
        print walls_matrix
   #     print "wall matrix size is: " + str(len(walls_matrix)) + " x " + str(len(walls_matrix[0]))
        walls = {}

        for y in range(len(walls_matrix)):
            for x in range(len(walls_matrix[0])):
                if walls_matrix[y][x] == 'x':
     #               print "found wall at " + str(x) + "," + str(y)
                    if y % 2 == 0:
     #                   print "this is a horizontal wall"
                        # horizontal walls are between the cur square the square to the right
                        walls = self.add_wall_edge(walls, x, y/2, x+1, y/2)

                    else:
        #                print "this is a vertical wall"
                        walls = self.add_wall_edge(walls, x, y/2, x, y/2 + 1)
        return walls_matrix

    def load_data(self, board_data, endx, endy):
        """
        We assume that there is no case
        when car be readed as vertical and horizontal at the same time
        Also we assume that there are no cars on the way of red car
        that can't be moved to side
        """
        results = self.str_to_matrix(board_data)
        matrix = results[0]
        self.wall_matrix = results[1]
        print matrix
        walls = self.generate_wall_dict(self.wall_matrix)
        self.generate_cars_horizontal(matrix)
        self.generate_cars_vertical(matrix)
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
            for direction in ['up', 'down', 'left', 'right']:
                if car.can_move(direction, 1, self.cars_to_matrix(cars)):
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
            moves, cars = Q.pop(0)

            if self.is_solved(cars):
                if len(moves) < min_moves:
                    is_doubled = False
                    min_moves = len(moves)
                    best_moves = deepcopy(moves)
                elif len(moves) == min_moves:
                    print(solver.format_steps(solver.cars, moves))
                    is_doubled = True

            for new_moves, new_cars in self.get_all_states(cars):
                if hash(str(new_cars)) not in visited:
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
        if red_car.orientation != ON_END: 
            return False
        if red_car.stop['x'] == self.endx and red_car.stop['y'] == self.endy:
            return True
        return False

    def format_steps(self, cars, moves):
        output = ''
        output += '\n\nSOLUTION\n'
        output += "; ".join(["{} {}".format(move[0], move[1]) for move in moves])
        cars = deepcopy(cars)
        for move in moves:
            car = tuple(filter(lambda x: x.character == move[0], cars))[0]
            output += '\nMOVE {} {}\n'.format(move[0], move[1])
            car.move(move[1], 1)
            output += self.format_data(cars)
        output += '\nEND of SOLUTION\n\n' + 'Total Moves: ' + str(len(moves))
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
            for x in range(len(self.wall_matrix[0])):
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
#   board_data = '''
#      BBB..C
#      AA..DC
#      ....DC
#      ErFFFF
#      E.....
#      E.....
#   '''

    wall_data = '''
       B|B|B|.|.|C
       - - - - - -
       A|A|.|.|D|C
       - - - x - -
       .|.|.|.|D|C
       - x - - - - 
       E|r|F|F|F|F
       - - - - - -
       Ex.|.|.|.|.
       - - - - - -
       E|.x.x.|.|.
    '''


    endx = 3
    endy = 0

    solver = Solver()
    solver.load_data(wall_data, endx, endy)
    print('Loaded data')
    solver.format_data(solver.cars)
    print('Looking for solution.. (may take several seconds)')
    moves = solver.solve()
    print(solver.format_steps(solver.cars, moves))
