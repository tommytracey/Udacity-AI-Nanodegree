assignments = []

rows = 'ABCDEFGHI'
cols = '123456789'


def cross(a, b):
    return [s+t for s in a for t in b]

# Create unit lists for rows, columns, and squares
boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
col_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

# Create diagnol unit lists
diag_units = [[rows[i]+cols[i] for i in range(len(rows))], [rows[::-1][i]+cols[i] for i in range(len(rows))]]

# Add diagnol units to the peers dictionary
unitlist = row_units + col_units + square_units + diag_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
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
        the values dictionary with the naked twins eliminated from peers.
    """
    # values = {"G7": "2345678", "G6": "1236789", "G5": "23456789", "G4": "345678", "G3": "1234569", "G2": "12345678", "G1": "23456789", "G9": "24578", "G8": "345678", "C9": "124578", "C8": "3456789", "C3": "1234569", "C2": "1234568", "C1": "2345689", "C7": "2345678", "C6": "236789", "C5": "23456789", "C4": "345678", "E5": "678", "E4": "2", "F1": "1", "F2": "24", "F3": "24", "F4": "9", "F5": "37", "F6": "37", "F7": "58", "F8": "58", "F9": "6", "B4": "345678", "B5": "23456789", "B6": "236789", "B7": "2345678", "B1": "2345689", "B2": "1234568", "B3": "1234569", "B8": "3456789", "B9": "124578", "I9": "9", "I8": "345678", "I1": "2345678", "I3": "23456", "I2": "2345678", "I5": "2345678", "I4": "345678", "I7": "1", "I6": "23678", "A1": "2345689", "A3": "7", "A2": "234568", "E9": "3", "A4": "34568", "A7": "234568", "A6": "23689", "A9": "2458", "A8": "345689", "E7": "9", "E6": "4", "E1": "567", "E3": "56", "E2": "567", "E8": "1", "A5": "1", "H8": "345678", "H9": "24578", "H2": "12345678", "H3": "1234569", "H1": "23456789", "H6": "1236789", "H7": "2345678", "H4": "345678", "H5": "23456789", "D8": "2", "D9": "47", "D6": "5", "D7": "47", "D4": "1", "D5": "36", "D2": "9", "D3": "8", "D1": "36"}

    # # Find all boxes with 2 digit values
    # possible_naked_twins = [box for box in boxes if len(values[box]) == 2]

    # # Find peers with matching values, aka naked twins
    # naked_twins = [[box_a, box_b] for box_a in possible_naked_twins \
    #                 for box_b in peers[box_a] \
    #                 if set(values[box_a]) == set(values[box_b])]
    # # print("naked_twins: ", naked_twins)

    # # Remove naked twin digits from common peers
    # for box_a, box_b in naked_twins:
    #     value = values[box_a]
    #     peers_common = set(peers[box_a]) & set(peers[box_b])
    #     for peer in peers_common:
    #         if len(values[peer]) > 2:
    #             for digit in value:
    #                 values = assign_value(values, peer, values[peer].replace(digit, ''))
                            
    # return values


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
    # all_digits = '123456789'
    # values = []

    # # Convert grid into a list of numbered character strings
    # for char in grid:
    #     if char == '.':
    #         values.append(all_digits)
    #     elif char in all_digits:
    #         values.append(char)
    
    # assert len(values) == 81
    
    # # add pairs to dictionary
    # grid_dict = dict(zip(boxes, values))
    grid_dict = {"G7": "2345678", "G6": "1236789", "G5": "23456789", "G4": "345678", "G3": "1234569", "G2": "12345678", "G1": "23456789", "G9": "24578", "G8": "345678", "C9": "124578", "C8": "3456789", "C3": "1234569", "C2": "1234568", "C1": "2345689", "C7": "2345678", "C6": "236789", "C5": "23456789", "C4": "345678", "E5": "678", "E4": "2", "F1": "1", "F2": "24", "F3": "24", "F4": "9", "F5": "37", "F6": "37", "F7": "58", "F8": "58", "F9": "6", "B4": "345678", "B5": "23456789", "B6": "236789", "B7": "2345678", "B1": "2345689", "B2": "1234568", "B3": "1234569", "B8": "3456789", "B9": "124578", "I9": "9", "I8": "345678", "I1": "2345678", "I3": "23456", "I2": "2345678", "I5": "2345678", "I4": "345678", "I7": "1", "I6": "23678", "A1": "2345689", "A3": "7", "A2": "234568", "E9": "3", "A4": "34568", "A7": "234568", "A6": "23689", "A9": "2458", "A8": "345689", "E7": "9", "E6": "4", "E1": "567", "E3": "56", "E2": "567", "E8": "1", "A5": "1", "H8": "345678", "H9": "24578", "H2": "12345678", "H3": "1234569", "H1": "23456789", "H6": "1236789", "H7": "2345678", "H4": "345678", "H5": "23456789", "D8": "2", "D9": "47", "D6": "5", "D7": "47", "D4": "1", "D5": "36", "D2": "9", "D3": "8", "D1": "36"}

    return grid_dict


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    # Copied from 'Strategy 1' lesson utils.py
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return


def eliminate(values):
    """Eliminates values from peers of each box with a single value.

    Goes through all the boxes, and whenever there is a box with a single value,
    eliminates this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    # Create list of solved boxes
    solved_boxes = [box for box in values.keys() if len(values[box]) == 1]
    
    # Eliminate solved values from peers 
    for box in solved_boxes:
        digit = values[box]
        for peer in peers[box]:
            # values[peer] = values[peer].replace(digit, '')
            values = assign_value(values, peer, values[peer].replace(digit,''))
    
    return values


def only_choice(values):
    """Finalizes all values that are the only choice for a unit.

    Goes through all the units, and whenever there is a unit with a value
    that only fits in one box, assigns the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for unit in unitlist:
        for digit in '123456789':
            d_boxes = [box for box in unit if digit in values[box]]
            if len(d_boxes) == 1:
                values[d_boxes[0]] = digit
    return values


def reduce_puzzle(values):
    """Uses the 'eliminate' and 'only_choice' functions as initial strategies to solve 
    the puzzle, or at least reduce the number of empty boxes. 

    The function stops if the puzzle gets solved or quits if it stops making progress. 
    
    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after applying updates.
    """
    stalled = False
    while not stalled:
        # Check boxes for determined values
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Use the Eliminate strategy
        values = eliminate(values)
        # Use the Only Choice strategy
        values = only_choice(values)
        # # Use Naked Twins strategy
        # values = naked_twins(values)
        # Check again how many boxes have a determined value
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # Stop the loop if no new values added
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            empty_boxes = [box for box in values.keys() if len(values[box]) == 0]
            return False

    return values


def search(values):
    """Creates a tree of possibilities and traverses it using depth-first search (DFS) until 
    it finds a solution for the sudoku puzzle.
    """
    # Reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all(len(values[box]) == 1 for box in boxes):
        print("\nSOLVED!\n")
        display(values)    
        return values
        
    # Choose one of the unsolved squares with the fewest possibilities
    count, box = min((len(values[box]), box) for box in boxes if len(values[box]) > 1)
    
    # Recurse tree of resulting sudokus; if one returns a value (not False), return that answer
    for digit in values[box]:
        new_board = values.copy()
        new_board[box] = digit
        attempt = search(new_board)
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
    # diag_sudoku_grid = '9.1....8.8.5.7..4.2.4....6...7......5..............83.3..6......9................'
    # diag_sudoku_grid = '8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..'
    # diag_sudoku_grid =  '...............9..97.3......1..6.5....47.8..2.....2..6.31..4......8..167.87......'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
