assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

# define boxes
boxes = cross(rows, cols)
#define row units
row_units = [cross(r, cols) for r in rows]
#define column units
column_units = [cross(rows, c) for c in cols]
#define square units
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
# Define diagonal units
diagonal_units = [[x+y for x, y in zip(rows, cols)], [x+y for x, y in zip(rows, cols[::-1])]]

#unlist would have all the constraints including row, column, square, diagonal
unitlist = row_units + column_units + square_units + diagonal_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers
    """

    # Find all instances of naked twins  
    for unit in unitlist:
        possible_twins = {}
        # for all possible values like (11,12,...98,99)in the grid find the possible twin values 
        for pair in cross(cols, cols):
            possible_twins[pair] = [box for box in unit if values[box] == pair]
            #print('values[box]',possible_twins[pair] )
        for x, y in possible_twins.items() :
            # if there are only two items then replace the naked twins
            if (len(y) == 2):
                #print('x,y',y)
                for box in unit:
                # replace the cells that can't contain the naked unit
                    if (box not in y):
                        values = assign_value(values, box, values[box].replace(x[0],''))
                        values = assign_value(values, box, values[box].replace(x[1],''))

    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    # chars = []
    # digits = '123456789'
    # for c in grid:
    #     if c in digits:
    #         chars.append(c)
    #     if c == '.':
    #         chars.append(digits)
    # assert len(chars) == 81
    # return dict(zip(boxes, chars))

    grid_dict = {"G7": "2345678", "G6": "1236789", "G5": "23456789", "G4": "345678", "G3": "1234569", "G2": "12345678", "G1": "23456789", "G9": "24578", "G8": "345678", "C9": "124578", "C8": "3456789", "C3": "1234569", "C2": "1234568", "C1": "2345689", "C7": "2345678", "C6": "236789", "C5": "23456789", "C4": "345678", "E5": "678", "E4": "2", "F1": "1", "F2": "24", "F3": "24", "F4": "9", "F5": "37", "F6": "37", "F7": "58", "F8": "58", "F9": "6", "B4": "345678", "B5": "23456789", "B6": "236789", "B7": "2345678", "B1": "2345689", "B2": "1234568", "B3": "1234569", "B8": "3456789", "B9": "124578", "I9": "9", "I8": "345678", "I1": "2345678", "I3": "23456", "I2": "2345678", "I5": "2345678", "I4": "345678", "I7": "1", "I6": "23678", "A1": "2345689", "A3": "7", "A2": "234568", "E9": "3", "A4": "34568", "A7": "234568", "A6": "23689", "A9": "2458", "A8": "345689", "E7": "9", "E6": "4", "E1": "567", "E3": "56", "E2": "567", "E8": "1", "A5": "1", "H8": "345678", "H9": "24578", "H2": "12345678", "H3": "1234569", "H1": "23456789", "H6": "1236789", "H7": "2345678", "H4": "345678", "H5": "23456789", "D8": "2", "D9": "47", "D6": "5", "D7": "47", "D4": "1", "D5": "36", "D2": "9", "D3": "8", "D1": "36"}
    return grid_dict

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return


def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values


def reduce_puzzle(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes): 
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and 
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    values = search(values)
    return values

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
