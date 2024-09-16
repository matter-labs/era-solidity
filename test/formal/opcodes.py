from z3 import BitVecVal, BV2Int, If, LShR, UDiv, ULT, UGT, URem, ZeroExt, Extract

def ADD(x, y):
	return x + y

def MUL(x, y):
	return x * y

def SUB(x, y):
	return x - y

def DIV(x, y):
	return If(y == 0, 0, UDiv(x, y))

def SDIV(x, y):
	return If(y == 0, 0, x / y)

def MOD(x, y):
	return If(y == 0, 0, URem(x, y))

def MULMOD(x, y, m):
	return If(m == 0, 0, Extract(x.size() - 1, 0, URem(ZeroExt(x.size(), x) * ZeroExt(x.size(), y), ZeroExt(m.size(), m))))

def ADDMOD(x, y, m):
	return If(m == 0, 0, Extract(x.size() - 1, 0, URem(ZeroExt(1, x) + ZeroExt(1, y), ZeroExt(1, m))))

def SMOD(x, y):
	return If(
		y == 0,
		0,
		If(
			x >= 0,
			If(y >= 0, x % y, x % (-y)),
			If(y >= 0, -((-x) % y), -((-x) % (-y)))
		)
	)

def LT(x, y):
	return If(ULT(x, y), BitVecVal(1, x.size()), BitVecVal(0, x.size()))

def GT(x, y):
	return If(UGT(x, y), BitVecVal(1, x.size()), BitVecVal(0, x.size()))

def SLT(x, y):
	return If(x < y, BitVecVal(1, x.size()), BitVecVal(0, x.size()))

def SGT(x, y):
	return If(x > y, BitVecVal(1, x.size()), BitVecVal(0, x.size()))

def EQ(x, y):
	return If(x == y, BitVecVal(1, x.size()), BitVecVal(0, x.size()))

def ISZERO(x):
	return If(x == 0, BitVecVal(1, x.size()), BitVecVal(0, x.size()))

def AND(x, y):
	return x & y

def OR(x, y):
	return x | y

def NOT(x):
	return ~(x)

def SHL(x, y):
	return y << x

def SHR(x, y):
	return LShR(y, x)

def SAR(x, y):
	return y >> x

def BYTE(i, x):
	bit = (i + 1) * 8
	return If(
		UGT(i, x.size() / 8 - 1),
		BitVecVal(0, x.size()),
		(LShR(x, (x.size() - bit))) & 0xff
	)

def SIGNEXTEND(i, x):
	bitBV = i * 8 + 7
	bitInt = BV2Int(i) * 8 + 7
	test = BitVecVal(1, x.size()) << bitBV
	mask = test - 1
	return If(
		bitInt >= x.size(),
		x,
		If(
			(x & test) == 0,
			x & mask,
			x | ~mask
		)
	)
