from __future__ import annotations

import numpy as np

from dataclasses import dataclass
import time
import itertools
from collections import Counter
from typing import List, Set, Tuple


# Sleep after importing numpy as it effects benchmarking
time.sleep(1)


# Type alias created to indicate the
# array used as input represents a sudoku
Sudoku = np.ndarray


@dataclass
class Link:
    """
    Represents a 1 in the matrix to be solved.
    Forms part of two circular doubly linked lists
    """

    __slots__ = ("left", "right", "up", "down", "col", "row")

    left: Link | None
    right: Link | None
    up: Link | None
    down: Link | None
    col: int
    row: int


class Matrix:
    """
    Stores references to the links in the matrix and manipulates them.
    """

    __slots__ = ("cols", "rows", "col_lengths")

    def __init__(self, cols: List[Link | None], rows: List[Link | None],
                 col_lengths: List[int]) -> None:

        self.cols = cols
        self.rows = rows
        self.col_lengths = col_lengths

    def remove(self, link: Link, removed: List[Link]) -> None:
        """
        Disconnects a link from the links above and below it.

        :param link: The link to be removed
        :param removed: A list that stores the removed links,
                        so they can be added again.
        :return: None
        """

        link.up.down = link.down
        link.down.up = link.up
        removed.append(link)

        c = link.col
        if link is self.cols[c]:
            self.cols[c] = link.down

        self.col_lengths[c] -= 1

    def add(self, link: Link) -> None:
        """
        Reconnects a link that has been removed.

        :param link: The link to be added
        :return: None
        """

        link.up.down = link
        link.down.up = link
        self.col_lengths[link.col] += 1

    def remove_row(self, start: Link, removed: List[Link]) -> None:
        """
        Removes all the links in a row. The start is excluded as
        the column will be removed from allowed_cols, so it won't be
        accessed directly and won't be accessible indirectly as
        everything connecting to it has been removed.

        :param start: A link in the row
        :param removed: A list that stores the removed links,
                        so they can be added again.
        :return: None
        """

        link = start
        self.remove(link, removed)
        while link.right is not start:
            link = link.right
            self.remove(link, removed)

    def remove_col(self, link: Link, removed: List[Link]) -> None:
        """
        Removes all the rows that have a 1 in a column. The start is
        excluded as the column will be removed from allowed_cols, so
        it won't be accessed directly and won't be accessible indirectly
        as everything connecting to it has been removed.

        :param link: A link in the column
        :param removed: A list that stores the removed links,
                        so they can be added again.
        :return: None
        """

        stop = link.up
        next_link = link.down
        while link is not stop:
            link = next_link
            next_link = link.down
            self.remove_row(link, removed)

    def choose_row(self, link: Link, removed: List[Link],
                   removed_cols: List[int], allowed_cols: Set[int]) -> None:
        """
        Removes all the rows connected to the columns with 1s in this row.

        :param link: A link in the row
        :param removed: A list that stores the removed links,
                        so they can be added again.
        :param removed_cols: The columns removed from allowed_cols
        :param allowed_cols: The set of columns the algorithm can still check
        :return: None
        """

        stop = link.left
        next_link = link.right
        self.remove_col(link, removed)
        allowed_cols.remove(link.col)
        removed_cols.append(link.col)
        while link is not stop:
            link = next_link
            next_link = link.right
            self.remove_col(link, removed)
            allowed_cols.remove(link.col)
            removed_cols.append(link.col)

    def get_rows(self, col: int) -> List[Link]:
        """
        Returns a list of links in every row with 1s in a column.

        :param col: The column where rows have 1
        :return: A list of links in every row of the column
        """

        start = self.cols[col]
        rows = []
        link = start
        while True:
            rows.append(link)
            if link.down is start:
                break
            link = link.down

        return rows

    def undo(self, removed: List[Link], removed_cols: List[int], allowed_cols: Set[int]) -> None:
        """
        Reverses the changes previously made to the matrix.

        :param removed: removed: A list that stores the removed links,
                        so they can be added again.
        :param removed_cols: The columns removed from allowed_cols
        :param allowed_cols: The set of columns the algorithm can still check
        :return: None
        """

        for link in reversed(removed):
            self.add(link)

        allowed_cols.update(removed_cols)


# Create the matrix as a numpy array (inefficient but not done as part of
# sudoku_solver, so it doesn't affect solution times.

matrix = np.zeros((729, 324), dtype=np.int64)

for row in range(9):
    for col in range(9):
        for val in range(9):
            # cells
            matrix[row * 81 + col * 9 + val, row * 9 + col] = 1

            # rows
            matrix[row * 81 + col * 9 + val, 81 + col * 9 + val] = 1

            # columns
            matrix[row * 81 + col * 9 + val, 162 + row * 9 + val] = 1

            # boxes
            matrix[row * 81 + col * 9 + val, 243 + 9 * val + (3 * (row // 3) + (col // 3))] = 1

# Links are created in order of column first as adjacent links in a column are often
# accessed together by Matrix.remove. This significantly improves the performance on
# my machine and is necessary makes the pre-generated version faster than the other version.

lookup = dict()

cols = []
col_lengths = []

for col in range(324):
    col_list = []
    for row in matrix[:, col].nonzero()[0]:
        col_list.append(Link(None, None, None, None, col, row))
        lookup[(row, col)] = col_list[-1]

    for i in range(len(col_list) - 1):
        col_list[i].down = col_list[i + 1]
        col_list[i + 1].up = col_list[i]

    cols.append(col_list[0])
    col_lengths.append(len(col_list))

    col_list[0].up = col_list[-1]
    col_list[-1].down = col_list[0]

rows = []

for row in range(729):
    row_list = []
    for col in matrix[row].nonzero()[0]:
        row_list.append(lookup[(row, col)])

    for i in range(len(row_list) - 1):
        row_list[i].right = row_list[i + 1]
        row_list[i + 1].left = row_list[i]

    rows.append(row_list[0])

    row_list[0].left = row_list[-1]
    row_list[-1].right = row_list[0]

allowed_cols = set(range(324))

WRONG = np.full((9, 9), -1)

m = Matrix(cols, rows, col_lengths)


def check_duplicates(arr: np.ndarray) -> bool:
    """
    Checks if there are duplicated elements other than 0.

    :param arr: The array to be checked
    :return: True if there are duplicated elements other than 0
    """

    c = Counter(arr)
    for k, v in c.items():
        if v > 1 and k != 0:
            return True
    return False


def verify(sudoku: Sudoku) -> bool:
    """
    Checks if a sudoku has more than one of a number in
    a row, column or box.

    :param sudoku: The sudoku
    :return: True if the sudoku appears valid
    """

    for i in range(9):
        if check_duplicates(sudoku[i]):
            return False

        if check_duplicates(sudoku[:, i]):
            return False

    for a, b in itertools.product(range(0, 9, 3), range(0, 9, 3)):
        if check_duplicates(sudoku[a:a + 3, b:b + 3].ravel()):
            return False

    return True


easy_cols = [None] * 324
easy_rows = [None] * 729


def add_row(row: int, col: int, val: int, easy_col_lengths: List[int],
            previous: List[Link | None], starts: List[Link | None],
            allowed_cols: Set[int]) -> None:
    """
    Adds a row to the matrix corresponding with a row, column and value.

    :param row: The corresponding row of the sudoku
    :param col: The corresponding column of the sudoku
    :param val: The corresponding value in the sudoku
    :param easy_col_lengths: The list of column lengths
    :param previous: The previous links added in each column
    :param starts: The first links added in each column
    :param allowed_cols: The set of columns the solver can select
    :return: None
    """

    link_row = row * 81 + col * 9 + val

    # cell
    link_col = row * 9 + col
    l1 = Link(None, None, previous[link_col], None, link_col, link_row)
    if starts[link_col] is None:
        starts[link_col] = l1
        easy_cols[link_col] = l1
    else:
        previous[link_col].down = l1
    previous[link_col] = l1
    easy_col_lengths[link_col] += 1

    easy_rows[link_row] = l1

    allowed_cols.add(link_col)

    # row
    link_col = 81 + col * 9 + val
    l2 = Link(l1, None, previous[link_col], None, link_col, link_row)
    if starts[link_col] is None:
        starts[link_col] = l2
        easy_cols[link_col] = l2
    else:
        previous[link_col].down = l2
    previous[link_col] = l2
    easy_col_lengths[link_col] += 1

    allowed_cols.add(link_col)

    # column
    link_col = 162 + row * 9 + val
    l3 = Link(l2, None, previous[link_col], None, link_col, link_row)
    if starts[link_col] is None:
        starts[link_col] = l3
        easy_cols[link_col] = l3
    else:
        previous[link_col].down = l3
    previous[link_col] = l3
    easy_col_lengths[link_col] += 1

    allowed_cols.add(link_col)

    # box
    link_col = 243 + 9 * val + (3 * (row // 3) + (col // 3))
    l4 = Link(l3, l1, previous[link_col], None, link_col, link_row)
    if starts[link_col] is None:
        starts[link_col] = l4
        easy_cols[link_col] = l4
    else:
        previous[link_col].down = l4
    previous[link_col] = l4
    easy_col_lengths[link_col] += 1

    allowed_cols.add(link_col)

    l1.left = l4
    l1.right = l2
    l2.right = l3
    l3.right = l4


def create_matrix(sudoku: Sudoku) -> Tuple[bool, Matrix | None, Set[int] | None]:
    """
    Converts a sudoku into the correct matrix representation.
    Only used by easy_sudoku_solver

    :param sudoku: The sudoku being represented
    :return: If a valid matrix could be generated,
             the matrix and columns still to be selected
    """

    global easy_cols
    global easy_rows

    easy_cols = [None] * 324
    easy_rows = [None] * 729

    easy_col_lengths = [0] * 324
    previous = [None] * 324
    starts = [None] * 324

    allowed_cols = set()

    col_sets = []
    for col in range(9):
        col_sets.append(set(sudoku[:, col]))

    box_sets = []

    for row, col in itertools.product(range(0, 9, 3), range(0, 9, 3)):
        box_sets.append(set(sudoku[row:row+3, col:col+3].ravel()))

    for row in range(9):
        vals = set(range(1, 10))
        vals -= set(sudoku[row])
        for col in range(9):
            if sudoku[row, col]:
                continue

            reduced_vals = vals - col_sets[col]
            reduced_vals -= box_sets[3 * (row // 3) + (col // 3)]

            if not reduced_vals:
                return False, None, None
            for val in reduced_vals:
                add_row(row, col, val - 1, easy_col_lengths, previous, starts, allowed_cols)

    for first, last in zip(starts, previous):
        if first is not None:
            first.up = last
            last.down = first

    return True, Matrix(easy_cols, easy_rows, easy_col_lengths), allowed_cols


def hard_sudoku_solver(sudoku: Sudoku) -> Sudoku:
    """
    Solves a sudoku. Optimised for hard sudokus by using
    a pre-generated matrix. This is worse for easy sudokus
    as the matrix needs to be updated before beginning solving
    and needs to be reset to its original state which
    outweighs the time saved by not creating the matrix and
    by the matrix being laid out to more efficiently use
    cache for easy sudokus.

    :param sudoku: The sudoku to be solved
    :return: The solved sudoku
    """

    removed = []
    removed_cols = []

    for x, y in zip(*sudoku.nonzero()):
        row = x * 81 + y * 9 + sudoku[x, y] - 1
        m.choose_row(rows[row], removed, removed_cols, allowed_cols)

    s = []
    if hard_solve(s, m):
        for val in s:
            sudoku[val // 81, val % 81 // 9] = val % 9 + 1
        m.undo(removed, removed_cols, allowed_cols)
        return sudoku
    else:
        m.undo(removed, removed_cols, allowed_cols)
        return WRONG


def easy_sudoku_solver(sudoku: Sudoku) -> Sudoku:
    """
    Solves a sudoku. Optimised for easy sudokus by generating
    a matrix for each call. This is better for easy sudokus
    as it avoids performing updates before beginning solving
    and resetting the matrix to its original state which
    outweigh the time saved by not creating the matrix and
    by the matrix being laid out to more efficiently use cache.

    :param sudoku: The sudoku to be solved
    :return: The solved sudoku
    """

    valid, m, allowed_cols = create_matrix(sudoku)

    if not valid:
        return WRONG

    s = []
    if easy_solve(s, m, allowed_cols):
        for val in s:
            sudoku[val // 81, val % 81 // 9] = val % 9 + 1
        return sudoku
    else:
        return WRONG


def hard_solve(solution: List[int], m: Matrix) -> bool:
    """
    Solves a sudoku represented as a matrix and returns the
    matrix to its original state. Doesn't check if the
    initial layout is valid.

    :param solution: The solution so far
    :param m: The matrix representing the sudoku
    :return: True if the sudoku could be solved
    """

    if not allowed_cols:
        return True

    col = min(allowed_cols, key=col_lengths.__getitem__)
    if col_lengths[col] == 0:
        return False

    for row in m.get_rows(col):
        solution.append(row.row)
        removed = []
        removed_cols = []
        m.choose_row(row, removed, removed_cols, allowed_cols)
        if hard_solve(solution, m):
            m.undo(removed, removed_cols, allowed_cols)
            return True
        m.undo(removed, removed_cols, allowed_cols)
        solution.pop(-1)

    return False


def easy_solve(solution: List[int], m: Matrix, allowed_cols: Set[int]) -> bool:
    """
    Solves a sudoku represented as a matrix and doesn't
    return the matrix to its original state. Doesn't check if
    the initial layout is valid.

    :param solution: The solution so far
    :param m: The matrix representing the sudoku
    :param allowed_cols: The columns still to be
                         selected by the solver
    :return: True if the sudoku could be solved
    """

    if not allowed_cols:
        return True

    col = min(allowed_cols, key=m.col_lengths.__getitem__)
    if m.col_lengths[col] == 0:
        return False

    for row in m.get_rows(col):
        solution.append(row.row)
        removed = []
        removed_cols = []
        m.choose_row(row, removed, removed_cols, allowed_cols)
        if easy_solve(solution, m, allowed_cols):
            return True
        m.undo(removed, removed_cols, allowed_cols)
        solution.pop(-1)

    return False


def sudoku_solver(sudoku: Sudoku) -> Sudoku:
    """
    Solves a sudoku. Selects if hard_sudoku_solver or
    easy_sudoku_solver should be used. Does check if
    the initial layout is valid

    :param sudoku: The sudoku to be solved
    :return: The solved sudoku
    """

    if not verify(sudoku):
        return WRONG

    if np.count_nonzero(sudoku) <= 40:
        return hard_sudoku_solver(sudoku)
    else:
        return easy_sudoku_solver(sudoku)
