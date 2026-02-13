import itertools
import random


class Minesweeper():
    """
    Minesweeper game handling
    """

    def __init__(self, height=8, width=8, mines=8):
        self.height = height
        self.width = width
        self.mines = set()
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        while len(self.mines) < mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        self.mines_found = set()

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        count = 0
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) == cell:
                    continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1
        return count


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells, and a count of how many are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        # If number of cells matches the count, all are definitely mines
        if len(self.cells) == self.count and self.count != 0:
            return self.cells
        return set()

    def known_safes(self):
        # If count is 0, all cells are definitely safe
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        # Remove mine from sentence and decrement count
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        # Remove safe cell from sentence, count remains same
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.moves_made = set()
        self.mines = set()
        self.safes = set()
        self.knowledge = []

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        # 1. Mark the cell as a move made
        self.moves_made.add(cell)

        # 2. Mark the cell as safe
        self.mark_safe(cell)

        # 3. Add a new sentence to the AI's knowledge base
        neighbors = set()
        i, j = cell
        for r in range(i - 1, i + 2):
            for c in range(j - 1, j + 2):
                if (r, c) == cell:
                    continue
                if 0 <= r < self.height and 0 <= c < self.width:
                    if (r, c) in self.mines:
                        count -= 1
                    elif (r, c) not in self.safes:
                        neighbors.add((r, c))

        new_sentence = Sentence(neighbors, count)
        self.knowledge.append(new_sentence)

        # 4 & 5. Infer new info and new sentences
        self.update_knowledge()

    def update_knowledge(self):
        """
        Loops through knowledge to draw inferences
        """
        changed = True
        while changed:
            changed = False

            # Find all known safes and mines across sentences
            new_safes = set()
            new_mines = set()
            for sentence in self.knowledge:
                new_safes.update(sentence.known_safes())
                new_mines.update(sentence.known_mines())

            if new_safes:
                for safe in new_safes.copy():
                    if safe not in self.safes:
                        self.mark_safe(safe)
                        changed = True

            if new_mines:
                for mine in new_mines.copy():
                    if mine not in self.mines:
                        self.mark_mine(mine)
                        changed = True

            # Remove empty sentences
            self.knowledge = [s for s in self.knowledge if s.cells]

            # Subset inference
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 != s2 and s1.cells.issubset(s2.cells):
                        diff_cells = s2.cells - s1.cells
                        diff_count = s2.count - s1.count
                        new_inf = Sentence(diff_cells, diff_count)
                        if new_inf not in self.knowledge:
                            self.knowledge.append(new_inf)
                            changed = True

    def make_safe_move(self):
        for move in self.safes:
            if move not in self.moves_made:
                return move
        return None

    def make_random_move(self):
        choices = []
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.moves_made and (i, j) not in self.mines:
                    choices.append((i, j))

        if not choices:
            return None
        return random.choice(choices)
