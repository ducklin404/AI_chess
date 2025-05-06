import random
from magic_bitboard import *


# ---------------------------------------------------------------------------#
# 6.  Public attack functions – branch-free
# ---------------------------------------------------------------------------#
def rook_attacks(sq: int, all_occ: int) -> int:
    occ = all_occ & ROOK_MASKS[sq]
    key = ((occ * ROOK_MAGIC[sq]) & FULL) >> ROOK_SHIFT[sq]
    return ROOK_TABLE[sq][key]

def bishop_attacks(sq: int, all_occ: int) -> int:
    occ = all_occ & BISH_MASKS[sq]
    key = ((occ * BISH_MAGIC[sq]) & FULL) >> BISH_SHIFT[sq]
    return BISH_TABLE[sq][key]

def queen_attacks(sq: int, all_occ: int) -> int:
    return rook_attacks(sq, all_occ) | bishop_attacks(sq, all_occ)


def knight_attacks(square: int) -> int:
    rank, file = divmod(square, 8)
    bitboard = 0

    for dr, df in ((2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)):
        r, f = rank + dr, file + df
        if 0 <= r < 8 and 0 <= f < 8:         
            bitboard |= 1 << (r*8 + f)          

    return bitboard


def king_attacks(square: int) -> int:
    rank, file = divmod(square, 8)
    bitboard = 0

    for dr, df in ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)):
        r, f = rank + dr, file + df
        if 0 <= r < 8 and 0 <= f < 8:         
            bitboard |= 1 << (r*8 + f)          

    return bitboard

def pawn_attacks(sq: int, white: bool) -> int:
    r, f = divmod(sq, 8)
    res = 0
    dir = 1 if white else -1                  
    for df in (-1, 1):
        nr, nf = r + dir, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            res |= 1 << (nr * 8 + nf)
    return res

KNIGHT_ATTACKS = [knight_attacks(sq)  for sq in range(64)]
KING_ATTACKS   = [king_attacks(sq)    for sq in range(64)]
PAWN_ATTACKS_W = [pawn_attacks(sq, 1) for sq in range(64)]
PAWN_ATTACKS_B = [pawn_attacks(sq, 0) for sq in range(64)]