class Matrix():
    def __init__(self, values=(2, 2, 0)):
        if type(values) is tuple:
            fill = values[2] if len(values) == 3 else 0
            self.m = [[fill for j in range(values[1])] for i in range(values[0])]
        elif type(values) is list:
            if type(values[0]) is not list:
                values = [values]
            values = [values[i][:] for i in range(len(values))]
            self.m = values
        else:
            raise BaseException("Wrong fill argument")

    def rows(self):
        return len(self.m)

    def cols(self):
        return len(self.m[0])

    @staticmethod
    def dot(L, R, bias=0):
        if isinstance(R, (int, float, complex, bool)):
            Mm = [[L.m[row][col] * R + bias for col in range(L.cols())]
                  for row in range(L.rows())]
            M = Matrix(Mm)
            return M
        # rows = L.rows()
        # cols = L.cols()
        if L.cols() != R.rows():
            raise BaseException("Invalid matrix dimensions!")
        M = Matrix(values=(R.rows(), L.cols()))
        for i in range(R.cols()):
            s = 0
            for j in range(L.rows()):
                # TODO: DOESNT WORK
                s += L[i][j] * R[j][i]
            M[i][j] = s

    def __mul__(self, rhs):
        if not isinstance(rhs, Matrix):
            M = Matrix(self.m)
            for row in range(self.rows()):
                for col in range(self.cols()):
                    M.m[row][col] *= rhs
        else:
            rows = self.rows()
            if self.cols() != rhs.rows():
                raise BaseException("Invalid matrix dimensions!")
            M = Matrix(values=(rows, rhs.cols()))
            for row in range(rows):
                for col in range(rhs.cols()):
                    s = 0
                    for i in range(self.cols()):
                        s += self.m[row][i] * rhs.m[i][col]
                    M.m[row][col] = s
        return M

    __rmul__ = __mul__

    # ~Matrix() - transposition overload
    def __invert__(self):
        Mm = [[self.m[j][i] for j in range(self.rows())] for i in range(self.cols())]
        return Matrix(Mm)

    @staticmethod
    def transpose(L):
        Mm = [[L.m[j][i] for j in range(L.rows())] for i in range(L.cols())]
        return Matrix(Mm)

    def __add__(self, rhs):
        M = Matrix(self.m)
        if not isinstance(rhs, Matrix):
            for row in range(self.rows()):
                for col in range(self.cols()):
                    M.m[row][col] += rhs
        else:
            if self.rows() != rhs.rows() or self.cols() != rhs.cols():
                raise BaseException("Invalid matrix dimensions!")
            for row in range(self.rows()):
                for col in range(self.cols()):
                    M.m[row][col] += rhs.m[row][col]
        return M

    def __sub__(self, rhs):
        M = Matrix(self.m)
        if not isinstance(rhs, Matrix):
            for row in range(self.rows()):
                for col in range(self.cols()):
                    M.m[row][col] -= rhs
        else:
            if self.rows() != rhs.rows() or self.cols() != rhs.cols():
                raise BaseException("Invalid matrix dimensions!")
            for row in range(self.rows()):
                for col in range(self.cols()):
                    M.m[row][col] -= rhs.m[row][col]
        return M

    # Entry-wise multiplication
    def HadamardProduct(self, rhs):
        if self.rows() != rhs.rows() or self.cols() != rhs.cols():
            raise BaseException("Invalid matrix dimensions!")
        M = Matrix(self.m)
        for row in range(self.rows()):
            for col in range(self.cols()):
                M.m[row][col] = self.m[row][col] * rhs.m[row][col]
        return M

    # Quite complicated
    # TODO: BETTER EXPLANATION
    def col_wise_mult(self, B):
        M = Matrix(self.m)
        for row in range(self.rows()):
            for col in range(self.cols()):
                M.m[row][col] = self.m[row][col] * B.m[0][col]
        return M

    def __pow__(self, rhs):
        M = Matrix(self.m)
        for row in range(self.rows()):
            for col in range(self.cols()):
                M.m[row][col] = M.m[row][col] ** rhs
        return M

    def __str__(self):
        s = ""
        rows = self.rows()
        cols = self.cols()
        for i in range(rows - 1):
            for j in range(cols):
                s += str(self.m[i][j]) + " "
            s += "\n"

        for j in range(cols):
            s += str(self.m[rows - 1][j]) + " "

        return s

    def __repr__(self):
        return "Matrix<{}, {}>".format(self.rows(), self.cols())

    def __round__(self):
        M = Matrix(self.m)
        for row in range(M.rows()):
            for col in range(M.cols()):
                M.m[row][col] = round(M.m[row][col])
        return M


# M = Matrix([[1, 2, 3], [4, 5, 6]])
# M1 = Matrix((2, 3, 2))
# print(M)
# print(M1)
# print(M - M1)
# print(M)
# print(M1)
