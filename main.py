from itertools import chain, combinations

import numpy as np
import pygame
import numpy
from scipy import ndimage
from functools import reduce
from operator import add

### Globals ###

pygame.init()

adj = [[0, 0], [0, -1], [-1, 0], [0, 1], [1, 0]]

TILE_HEIGHT = 50
TILE_WIDTH = 50
MARGIN = 2


class Game:
    def __init__(self, cells):
        self.cells = cells
        self.clear()
        self.load_level()

    def clear(self):
        self.grid = [[0 for i in range(len(self.cells))] for j in range(len(self.cells))]

    def load_level(self):
        for y in range(len(self.cells)):
            for x in range(len(self.cells[y])):
                self.grid[x][y] = int(self.cells[y][x])

    def draw(self):
        for y in range(len(self.cells)):
            for x in range(len(self.cells)):
                i = x * TILE_WIDTH + MARGIN
                j = y * TILE_HEIGHT + MARGIN
                h = TILE_HEIGHT - (2 * MARGIN)
                w = TILE_WIDTH - (2 * MARGIN)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(screen, (105, 210, 231), [i, j, w, h])
                else:
                    pygame.draw.rect(screen, (255, 255, 255), [i, j, w, h])

    def get_adjacent(self, x, y):
        adjs = []
        for i, j in adj:
            if (0 <= i + x < len(self.cells)) and (0 <= j + y < len(self.cells)):
                adjs += [[i + x, j + y]]
        return adjs

    def click(self, pos):
        x = int(pos[0] / TILE_WIDTH)
        y = int(pos[1] / TILE_HEIGHT)
        adjs = self.get_adjacent(x, y)
        for i, j in adjs:
            self.grid[j][i] = (self.grid[j][i] + 1) % 2


class GF2(object):

    def __init__(self, a=0):
        self.value = int(a) & 1

    def __add__(self, rhs):
        return GF2(self.value + GF2(rhs).value)

    def __mul__(self, rhs):
        return GF2(self.value * GF2(rhs).value)

    def __sub__(self, rhs):
        return GF2(self.value - GF2(rhs).value)

    def __truediv__(self, rhs):
        return GF2(self.value / GF2(rhs).value)

    def __repr__(self):
        return str(self.value)

    def __eq__(self, rhs):
        if isinstance(rhs, GF2):
            return self.value == rhs.value
        return self.value == rhs

    def __le__(self, rhs):
        if isinstance(rhs, GF2):
            return self.value <= rhs.value
        return self.value <= rhs

    def __lt__(self, rhs):
        if isinstance(rhs, GF2):
            return self.value < rhs.value
        return self.value < rhs

    def __int__(self):
        return self.value

    def __long__(self):
        return self.value


GF2array = np.vectorize(GF2)


def gjel(A):
    nulldim = 0
    for i, row1 in enumerate(A):
        pivot = A[i:, i].argmax() + i
        if A[pivot, i] == 0:
            nulldim = len(A) - i
            break
        new_row = A[pivot] / A[pivot, i]
        A[pivot] = A[i]
        row1[:] = new_row

        for j, row2 in enumerate(A):
            if j == i:
                continue
            row2[:] -= new_row * A[j, i]
    return A, nulldim


def GF2inv(A):
    n = len(A)
    assert n == A.shape[1], "Matrix must be square"

    A = np.hstack([A, np.eye(n)])
    B, nulldim = gjel(GF2array(A))

    inverse = np.int_(B[-n:, -n:])
    E = B[:n, :n]
    null_vectors = []
    if nulldim > 0:
        null_vectors = E[:, -nulldim:]
        null_vectors[-nulldim:, :] = GF2array(np.eye(nulldim))
        null_vectors = np.int_(null_vectors.T)

    return inverse, null_vectors


def lightsoutbase(n):
    a = np.eye(n * n)
    a = np.reshape(a, (n * n, n, n))
    a = np.array(list(map(ndimage.binary_dilation, a)))
    return np.reshape(a, (n * n, n * n))


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class LightsOut(object):

    def __init__(self, size=5):
        self.n = size
        self.base = lightsoutbase(self.n)
        self.invbase, self.null_vectors = GF2inv(self.base)

    def solve(self, b):
        b = np.asarray(b)
        assert b.shape[0] == b.shape[1] == self.n, "incompatible shape"

        if not self.issolvable(b):
            raise ValueError("The given setup is not solvable")

        first = np.dot(self.invbase, b.ravel()) & 1
        solved = []
        for i in range(len(first)):
            if first[i] == 1:
                solved.append(i+1)
        print(solved)
        # solutions = [(first + reduce(add, nvs, 0)) & 1 for nvs in powerset(self.null_vectors)]
        # final = min(solutions, key=lambda x: x.sum())
        return np.reshape(first, (self.n, self.n))

    def issolvable(self, b):
        b = np.asarray(b)
        assert b.shape[0] == b.shape[1] == self.n, "incompatible shape"
        b = b.ravel()
        p = [np.dot(x, b) & 1 for x in self.null_vectors]
        return not any(p)


### Main ###

if __name__ == "__main__":

    # lo = LightsOut(3)
    # cells = numpy.array([[1, 1, 0],
    #                      [0, 1, 0],
    #                      [0, 1, 1]])
    # lo = LightsOut(4)
    # cells = numpy.array([[1, 1, 0, 0],
    #                      [0, 1, 0, 1],
    #                      [0, 1, 1, 0],
    #                      [0, 1, 1, 1]])
    # lo = LightsOut(5)
    # cells = numpy.array([[1, 0, 0, 0, 1],
    #                      [0, 1, 0, 1, 0],
    #                      [0, 0, 1, 0, 0],
    #                      [0, 1, 0, 1, 0],
    #                      [1, 0, 0, 0, 1]])
    # lo = LightsOut(6)
    # cells = numpy.array([[1, 1, 0, 0, 1, 0],
    #                      [0, 1, 0, 1, 0, 1],
    #                      [0, 1, 1, 0, 0, 0],
    #                      [0, 1, 1, 1, 0, 0],
    #                      [1, 1, 1, 1, 1, 1],
    #                      [1, 1, 0, 0, 1, 0]])
    lo = LightsOut(7)
    cells = numpy.array([[1, 1, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0],
                         [1, 1, 1, 0, 1, 1, 1]])
    # lo = LightsOut(8)
    # cells = numpy.array([[1, 1, 0, 0, 1, 0, 1, 0],
    #                      [0, 1, 0, 1, 0, 1, 0, 1],
    #                      [0, 1, 1, 0, 0, 0, 0, 0],
    #                      [0, 1, 1, 1, 0, 0, 1, 0],
    #                      [1, 1, 1, 1, 1, 1, 1, 1],
    #                      [1, 1, 0, 0, 1, 0, 0, 0],
    #                      [1, 1, 1, 0, 1, 1, 1, 0],
    #                      [1, 0, 0, 1, 0, 1, 0, 1]])
    bsol = lo.solve(cells)
    print(bsol)

    screen = pygame.display.set_mode((len(cells) * TILE_WIDTH, len(cells) * TILE_HEIGHT))
    screen.fill((167, 219, 216))
    pygame.display.set_caption("Game")

    game = Game(cells.T)
    game.draw()

    clock = pygame.time.Clock()
    keepGoing = True
    while keepGoing:
        clock.tick(30)
        game.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.click(pos)
        pygame.display.flip()
    pygame.quit()