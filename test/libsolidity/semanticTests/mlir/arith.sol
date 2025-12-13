contract C {
  function uc(uint a, uint b) public returns (uint) {
    uint r = 0;
    unchecked {
    r += a + b;
    r *= a * b;
    r -= a - b;
    r = (a + r) * 2;
    r = r / 2;
    r /= 2;
    r = r % 13;
    r %= 13;
    r = a ** b;
    }
    return r;
  }

  function c(uint a, uint b) public returns (uint) {
    return a + b;
  }

  function ofa(uint8 a) public returns (uint8) {
    return a + 1;
  }

  function ofs(int8 a) public returns (int8) {
    return a - 1;
  }

  function ofm(int8 a) public returns (int8) {
    return a * 2;
  }

  function ofd(int8 a) public returns (int8) {
    return a / -1;
  }

  function dbz(int8 a) public returns (int8) {
    return a / 0;
  }

  function neg(int a) public returns (int) {
    return -a;
  }

  function inc(int a) public returns (int) {
    int r = a++;
    r += a;
    return ++r;
  }

  function dec(int8 a) public returns (int8) {
    return a--;
  }

  function shl(uint a, uint b) public returns (uint) {
    return a << b;
  }

  function shr(uint a, uint b) public returns (uint) {
    return a >> b;
  }

  function shl8(uint8 a, uint b) public returns (uint8) {
    return a << b;
  }

  function shr8(uint8 a, uint b) public returns (uint8) {
    return a >> b;
  }

  function bit(int a, int b) public returns (int, int, int) {
    return (a & b, a | b, a ^ b);
  }

  function bit8(int8 a, int8 b) public returns (int8, int8, int8) {
    return (a & b, a | b, a ^ b);
  }

  function expu256(uint256 a, uint256 b) public returns (uint256) {
    return a ** b;
  }

  function expu8(uint8 a, uint8 b) public returns (uint8) {
    return a ** b;
  }

  function exp256(int256 a, uint256 b) public returns (int256) {
    return a ** b;
  }

  function exp8(int8 a, uint8 b) public returns (int8) {
    return a ** b;
  }
}

// ====
// compileViaMlir: true
// ----
// uc(uint256,uint256): 4, 2 -> 16
// c(uint256,uint256): -2, 1 -> -1
// c(uint256,uint256): -2, 2 -> FAILURE, hex"4e487b71", 0x11
// ofa(uint8): 255 -> FAILURE, hex"4e487b71", 0x11
// ofs(int8): -128 -> FAILURE, hex"4e487b71", 0x11
// ofm(int8): 127 -> FAILURE, hex"4e487b71", 0x11
// ofd(int8): -128 -> FAILURE, hex"4e487b71", 0x11
// dbz(int8): 1 -> FAILURE, hex"4e487b71", 0x12
// neg(int256): 1 -> -1
// inc(int256): 0 -> 2
// dec(int8): -128 -> FAILURE, hex"4e487b71", 0x11
// shl(uint256,uint256): 4, 1 -> 8
// shr(uint256,uint256): 4, 1 -> 2
// shl8(uint8,uint256): 1, 8 -> 0
// shr8(uint8,uint256): 1, 8 -> 0
// bit(int256,int256): 6, 3 -> 2, 7, 5
// bit8(int8,int8): 6, 3 -> 2, 7, 5
// expu256(uint256,uint256): 3, 4 -> 81
// expu256(uint256,uint256): 0, 0 -> 1
// expu256(uint256,uint256): 7, 0 -> 1
// expu256(uint256,uint256): 0, 3 -> 0
// expu256(uint256,uint256): 2, 8 -> 256
// expu256(uint256,uint256): 1, 255 -> 1
// expu256(uint256,uint256): 2, 256 -> FAILURE, hex"4e487b71", 0x11
// expu256(uint256,uint256): 2, 255 -> 57896044618658097711785492504343953926634992332820282019728792003956564819968
// expu256(uint256,uint256): 10, 77 -> 100000000000000000000000000000000000000000000000000000000000000000000000000000
// expu256(uint256,uint256): 306, 31 -> 114120645878241659292862104872151745131076900847207153599812066301423392915456
// expu256(uint256,uint256): 306, 32 -> FAILURE, hex"4e487b71", 0x11
// expu256(uint256,uint256): 20, 40 -> 10995116277760000000000000000000000000000000000000000
// expu256(uint256,uint256): 20, 43 -> 87960930222080000000000000000000000000000000000000000000
// expu256(uint256,uint256): 307, 32 -> FAILURE, hex"4e487b71", 0x11
// expu8(uint8,uint8): 3, 4 -> 81
// expu8(uint8,uint8): 0, 0 -> 1
// expu8(uint8,uint8): 7, 0 -> 1
// expu8(uint8,uint8): 0, 3 -> 0
// expu8(uint8,uint8): 1, 255 -> 1
// expu8(uint8,uint8): 2, 8 -> FAILURE, hex"4e487b71", 0x11
// expu8(uint8,uint8): 3, 6 -> FAILURE, hex"4e487b71", 0x11
// expu8(uint8,uint8): 12, 33 -> FAILURE, hex"4e487b71", 0x11
// exp256(int256,uint256): 3, 4 -> 81
// exp256(int256,uint256): 0, 0 -> 1
// exp256(int256,uint256): 7, 0 -> 1
// exp256(int256,uint256): 0, 3 -> 0
// exp256(int256,uint256): 2, 8 -> 256
// exp256(int256,uint256): -7, 1 -> -7
// exp256(int256,uint256): -7, 3 -> -343
// exp256(int256,uint256): 300000000000000000000000000000000000000, 2 -> FAILURE, hex"4e487b71", 0x11
// exp256(int256,uint256): -300000000000000000000000000000000000000, 3 -> FAILURE, hex"4e487b71", 0x11
// exp256(int256,uint256): 2, 255 -> FAILURE, hex"4e487b71", 0x11
// exp256(int256,uint256): -2, 255 -> -57896044618658097711785492504343953926634992332820282019728792003956564819968
// exp256(int256,uint256): -2, 256 -> FAILURE, hex"4e487b71", 0x11
// exp8(int8,uint8): 3, 4 -> 81
// exp8(int8,uint8): -4, 3 -> -64
// exp8(int8,uint8): 0, 0 -> 1
// exp8(int8,uint8): 7, 0 -> 1
// exp8(int8,uint8): 0, 3 -> 0
// exp8(int8,uint8): 1, 255 -> 1
// exp8(int8,uint8): -1, 255 -> -1
// exp8(int8,uint8): 2, 7 -> FAILURE, hex"4e487b71", 0x11
// exp8(int8,uint8): -2, 7 -> -128
// exp8(int8,uint8): 3, 6 -> FAILURE, hex"4e487b71", 0x11
// exp8(int8,uint8): 16, 2 -> FAILURE, hex"4e487b71", 0x11
// exp8(int8,uint8): -16, 2 -> FAILURE, hex"4e487b71", 0x11
// exp8(int8,uint8): 3, 5 -> FAILURE, hex"4e487b71", 0x11
// exp8(int8,uint8): -3, 5 -> FAILURE, hex"4e487b71", 0x11
