from crossword import Variable, Crossword   # IMPORTANT: must be at module level
from itertools import product
import copy
import sys


class CrosswordCreator:

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: set(self.crossword.words)
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array of letters in crossword puzzle.
        """
        width = self.crossword.width
        height = self.crossword.height
        cells = [
            [None for _ in range(width)] for _ in range(height)
        ]
        for variable, word in assignment.items():
            d = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if d == Variable.DOWN else 0)
                j = variable.j + (k if d == Variable.ACROSS else 0)
                cells[i][j] = word[k]
        return cells

    def print(self, assignment):
        """
        Print crossword assignment to terminal.
        """
        cells = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(cells[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        cells = self.letter_grid(assignment)

        img = Image.new(
            "RGBA",
            (
                self.crossword.width * cell_size,
                self.crossword.height * cell_size
            ),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if cells[i][j]:
                        w, h = draw.textsize(cells[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10
                            ),
                            cells[i][j], fill="black", font=font
                        )
                else:
                    draw.rectangle(rect, fill="black", width=0)

        img.save(filename)

    def enforce_node_consistency(self):
        """
        Keep only words with the correct length for each variable.
        """
        for var in self.domains:
            to_keep = {word for word in self.domains[var] if len(word) == var.length}
            self.domains[var] = to_keep

    def revise(self, x, y):
        """
        Make variable x arc consistent with variable y.
        Returns True iff the domain of x was changed.
        """
        overlap = self.crossword.overlaps.get((x, y))
        if overlap is None:
            return False

        i, j = overlap
        revised = False
        to_remove = []

        for x_word in self.domains[x]:
            if not any(x_word[i] == y_word[j] for y_word in self.domains[y]):
                to_remove.append(x_word)
                revised = True

        for word in to_remove:
            self.domains[x].remove(word)

        return revised

    def ac3(self, arcs=None):
        """
        Enforce arc consistency using AC‑3.
        """
        queue = []
        if arcs is None:
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
        else:
            queue = list(arcs)

        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if every crossword variable is assigned a value.
        """
        for var in self.crossword.variables:
            if var not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Check if assignment satisfies:
          - all distinct words
          - correct word lengths
          - no conflicts with neighbours
        """
        values = list(assignment.values())
        if len(values) != len(set(values)):
            return False

        for var, word in assignment.items():
            if len(word) != var.length:
                return False

            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps.get((var, neighbor))
                    if overlap is not None:
                        i, j = overlap
                        if word[i] != assignment[neighbor][j]:
                            return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Least‑constraining‑values heuristic.
        Return values sorted by how many neighbour domain values they eliminate.
        """
        unassigned_neighbors = [
            n for n in self.crossword.neighbors(var)
            if n not in assignment
        ]

        if not unassigned_neighbors:
            return list(self.domains[var])

        def eliminated_count(value):
            count = 0
            for neighbor in unassigned_neighbors:
                overlap = self.crossword.overlaps.get((var, neighbor))
                if overlap is None:
                    continue
                i, j = overlap
                for n_word in self.domains[neighbor]:
                    if n_word[j] != value[i]:
                        count += 1
            return count

        return sorted(self.domains[var], key=eliminated_count)

    def select_unassigned_variable(self, assignment):
        """
        MRV + Degree heuristic.
        """
        unassigned = [v for v in self.crossword.variables if v not in assignment]

        def mrv_key(var):
            domain_size = len(self.domains[var])
            degree = len([
                n for n in self.crossword.neighbors(var)
                if n not in assignment
            ])
            return (domain_size, -degree)

        return min(unassigned, key=mrv_key)

    def backtrack(self, assignment):
        """
        Recursive backtracking search.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value

            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        return None

    def solve(self):
        """
        Enforce node and arc consistency, then run backtracking.
        Returns a complete assignment or None.
        """
        self.enforce_node_consistency()
        if not self.ac3():
            return None
        return self.backtrack(dict())


def main():
    # Parse command line
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output.png]")

    structure_file = sys.argv[1]
    words_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) == 4 else None

    # Create crossword & solver
    crossword = Crossword(structure_file, words_file)
    creator = CrosswordCreator(crossword)

    # Solve the puzzle (using our new method)
    assignment = creator.solve()

    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output_file:
            creator.save(assignment, output_file)


if __name__ == "__main__":
    main()
