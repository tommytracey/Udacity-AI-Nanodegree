assignments = []


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
        print("assignments: ", assignments)
    return values


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    print("\n START naked_twins")
    print("values in: ", values)

    # print("A1 value", values['A1'])

    ## Reduce or solve puzzle using the previous functions
    values = reduce_puzzle(values)
    # print("\nsearch values out: ", values)
    
    if values is False:
        print("\nno values!")
        return False
    
    # Check to see if puzzle is solved
    if all(len(values[box]) == 1 for box in boxes):
        print("\nsolved!")
        print("\nfinal values", values, "\n")
        display(values)
        return values

    # If not solved, find all 2 digit boxes within each square unit
    # print("\nsquare_units: ", square_units)

    # print("\nsample A1 value: ", values['A1'])

    for unit in unitlist:
        boxes_2d = []
        for box in unit:
            print("box, values[box]: {}, {}".format(box, values[box]))
            if len(values[box]) == 2:
                boxes_2d.append([box, values[box]])
                print("\nboxes_2d", boxes_2d)
                # check whether boxes contain naked twins
                if len(boxes_2d) > 1:
                    boxes_2d_sort = sorted(boxes_2d, key=lambda x: x[1])
                    print("\nboxes_2d_sort", boxes_2d_sort)
                    if boxes_2d_sort[0][1] == boxes_2d_sort[1][1]:
                        for box, value in boxes_2d_sort:
                            for digit in value:
                                new_board = values.copy()
                                new_board[box] = digit
                                attempt = naked_twins(new_board)
                                print("attempt", attempt)
                                if attempt:
                                    return attempt
                                else:
                                    # print("\nvalues out: ", values)
                                    return values

    values = search(values)


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]


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
    all_digits = '123456789'
    values = []

    # Convert grid into a list of numbered character strings
    for char in grid:
        if char == '.':
            values.append(all_digits)
        elif char in all_digits:
            values.append(char)

    # add pairs to dictionary
    grid_dict = dict(zip(boxes, values))
    print("orginal values: ", grid_dict)
    return grid_dict


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    # Borrowed from 'Strategy 1' lesson utils.py
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
    print("\n START eliminate()")
    print("values in: ", values)

    # Create list of solved boxes
    solved_boxes = [box for box in values.keys() if len(values[box]) == 1]
    
    # Check to see if E5 is solved
    for box in solved_boxes:
        if box == 'E5':
        # Remove E5 digit from peers
            remove_from_peers('E5', values)
    # Iterate through diaganols and remove solved digits from unsolved peers
        for unit in diag_units:
            if box in unit:
                remove_from_peers(box, values)
        remove_from_peers(box, values)

    ### ORIGINAL ### 
    # Iterate through other units and remove solved digits from unsolved peers
    
        # digit = values[box]
        # for peer in peers[box]:
        #     if len(values[peer]) > 1:
        #         values[peer] = values[peer].replace(digit, '')
    
    print("\neliminate values out: ", values)
    display(values)
    return values

def remove_from_peers(box, values):
    """Removes the digit within a solved box from the peers of that box.
    """
    digit = values[box]
    for peer in peers[box]:
        if len(values[peer]) > 1:
            if digit in values[peer]:
                print("{} to be removed from value {} for box {}".format(digit, values[peer], box))
                values[peer] = values[peer].replace(digit, '')
                print("new value is {}".format(values[peer]))

    return values


def only_choice(values):
    """Finalizes all values that are the only choice for a unit.

    Goes through all the units, and whenever there is a unit with a value
    that only fits in one box, assigns the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    # print("\n START only_choice()")
    # print("values in: ", values)

    for unit in unitlist:
        for digit in '123456789':
            d_boxes = [box for box in unit if digit in values[box]]
            if len(d_boxes) == 1:
                values[d_boxes[0]] = digit
    print("\nonly_choice values out: ", values)
    display(values)
    return values


def reduce_puzzle(values):
    """Uses the 'eliminate' and 'only_choice' functions as initial strategies to solve 
    the puzzle, or at least reduce the number of empty boxes. 

    The function stops if the puzzle gets solved or quits if it stops making progress. 
    
    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after applying updates.
    """
    print("\n START reduce_puzzle()")
    print("values in: ", values)

    stalled = False
    while not stalled:
        # Check boxes for determined values
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # # Use Naked Twins strategy
        # values = naked_twins(values)
        # Use the Eliminate strategy
        values = eliminate(values)
        # Use the Only Choice strategy
        values = only_choice(values)
        # Check again how many boxes have a determined value
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # Stop the loop if no new values added
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            print("line 192 returns False ")
            empty_boxes = [box for box in values.keys() if len(values[box]) == 0]
            print("empty_boxes: ", empty_boxes)
            return False

    print("\nreduce_puzzle values out: ", values)
    display(values)

    return values


def search(values):
    """Creates a tree of possibilities and traverses it using depth-first search (DFS) until 
    it finds a solution for the sudoku puzzle.
    """
    print("\n START search()")
    print("values in: ", values)

    # Reduce the puzzle using the previous function
    values = reduce_puzzle(values)

    if values is False:
        return False
    if all(len(values[box]) == 1 for box in boxes):
        display(values)    
        return values
        print("solved!")
        
    # Choose one of the unfilled squares with the fewest possibilities
    count, box = min((len(values[box]), box) for box in boxes if len(values[box]) > 1)
    
    # Recurse tree of resulting sudokus; if one returns a value (not False), return that answer!
    for digit in values[box]:
        new_board = values.copy()
        new_board[box] = digit
        attempt = search(new_board)
        if attempt:
            print("\nsearch values out (attempt): ", attempt)
            display(values)
            return attempt
        else:
            print("\nsearch values out: ", values)
            display(values)

            return values

    values = naked_twins(values)


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """

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


### HELPER ###


rows = 'ABCDEFGHI'
cols = '123456789'


def cross(a, b):
    return [s+t for s in a for t in b]


def diagonal(rows):
    col_a, col_b = 1, 9
    diag_a, diag_b = [], []
    for r in rows:
        box_a, box_b = r + str(col_a), r + str(col_b)
        diag_a.append(box_a)
        diag_b.append(box_b)
        col_a += 1
        col_b -= 1
    # print("[diag_a, diag_b] = ", [diag_a, diag_b])
    return [diag_a, diag_b]


boxes = cross(rows, cols)

# print(boxes)

row_units = [cross(r, cols) for r in rows]

# print("\nrow_units", row_units)

col_units = [cross(rows, c) for c in cols]

# print(col_units[0])

square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

# print(square_units[0])

diag_units = diagonal(rows)

# print("\ndiag_units", diag_units)

unitlist = row_units + col_units + square_units + diag_units

# print("\nunitlist", unitlist)

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)

# print("\nunits", units)

peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

# print("\npeers", peers)
# print("\npeers['A1']: ", peers['A1'])
# print("\npeers['A2']: ", peers['A2'])
# print("\npeers['E5']: ", peers['E5'])