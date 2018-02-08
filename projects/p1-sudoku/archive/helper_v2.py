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
	return [diag_a, diag_b]


boxes = cross(rows, cols)

# print(boxes)

row_units = [cross(r, cols) for r in rows]

print("\nrow_units", row_units)

col_units = [cross(rows, c) for c in cols]

# print(col_units[0])

square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

# print(square_units[0])

diag_units = diagonal(rows)

# print("\ndiag_units", diag_units)

unitlist = row_units + col_units + square_units + diag_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)

# print("\nunits", units)

peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

# print("\npeers", peers)