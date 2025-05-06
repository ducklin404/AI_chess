from __future__ import annotations
import random
import pickle
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------#
# 1.  Basic helpers
# ---------------------------------------------------------------------------#
FULL       = 0xFFFF_FFFF_FFFF_FFFF
FILE_A     = 0x0101_0101_0101_0101
FILE_H     = FILE_A << 7
RANK_1     = 0x0000_0000_0000_00FF
RANK_8     = RANK_1 << 56
DIRECTIONS = {
    "rook"  : (( 1,0),(-1,0),(0, 1),(0,-1)),
    "bishop": (( 1,1),( 1,-1),(-1,1),(-1,-1)),
}

def bit(r: int, f: int) -> int:
    return 1 << (r*8 + f)

def popcount(x: int) -> int:
    return x.bit_count()

# ---------------------------------------------------------------------------#
# 2.  Masks – squares on the same ray but *inside* the rim
# ---------------------------------------------------------------------------#
def rook_mask(sq: int) -> int:
    r, f = divmod(sq, 8)
    mask = 0
    for i in range(r+1, 7): mask |= bit(i, f)        
    for i in range(1, r):   mask |= bit(r-i, f)       
    for i in range(f+1, 7): mask |= bit(r, i)         
    for i in range(1, f):   mask |= bit(r, f-i)       
    return mask

def bishop_mask(sq: int) -> int:
    r, f = divmod(sq, 8)
    mask = 0
    for i in range(1, min(7-r, 7-f)): mask |= bit(r+i, f+i)  
    for i in range(1, min(r, f)):     mask |= bit(r-i, f-i)  
    for i in range(1, min(7-r, f)):   mask |= bit(r+i, f-i)  
    for i in range(1, min(r, 7-f)):   mask |= bit(r-i, f+i)   
    return mask

ROOK_MASKS  = [rook_mask(sq)   for sq in range(64)]
BISH_MASKS  = [bishop_mask(sq) for sq in range(64)]

# ---------------------------------------------------------------------------#
# 3.  On-the-fly ray walk (used during table generation only)
# ---------------------------------------------------------------------------#
def sliding_attacks(sq: int, occ: int, rook: bool) -> int:
    dirs = DIRECTIONS["rook"] if rook else DIRECTIONS["bishop"]
    r, f  = divmod(sq, 8)
    attacks = 0
    for dr, df in dirs:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            s = nr*8 + nf
            attacks |= 1 << s
            if occ & (1 << s):        # blocker?
                break
            nr += dr; nf += df
    return attacks

# ---------------------------------------------------------------------------#
# 4.  Magic-number search
# ---------------------------------------------------------------------------#
def random_magic() -> int:
    """Sparse random number - statistically good candidate."""
    return (random.getrandbits(64) &
            random.getrandbits(64) &
            random.getrandbits(64))

def find_magic(sq: int, mask: int, rook: bool) -> tuple[int, int, list[int]]:
    """Return (magic, shift, attack_table) for this square."""
    bits   = [i for i in range(64) if mask >> i & 1]  # positions of 1-bits
    n_bits = len(bits)
    table_size = 1 << n_bits
    shift  = 64 - n_bits

    # Enumerate every occupancy subset only once and cache results
    occs   = [0]*table_size
    atts   = [0]*table_size
    for idx in range(table_size):
        occ = 0
        for i, b in enumerate(bits):
            if idx >> i & 1:
                occ |= 1 << b
        occs[idx] = occ
        atts[idx] = sliding_attacks(sq, occ, rook)

    while True:
        magic = random_magic()
        used  = [-1]*table_size
        ok    = True
        for occ, att in zip(occs, atts):
            key = ((occ * magic) & FULL) >> shift
            if used[key] == -1:
                used[key] = att
            elif used[key] != att:
                ok = False
                break
        if ok:
            return magic, shift, used             # 'used' is the table itself

# ---------------------------------------------------------------------------#
# 5.  Build everything for all 64 squares 
# ---------------------------------------------------------------------------#
def build_magic_tables():
    rook_magic, rook_shift, rook_table = [], [], []
    bish_magic, bish_shift, bish_table = [], [], []

    for sq in range(64):
        m, s, t = find_magic(sq, ROOK_MASKS[sq], rook=True)
        rook_magic.append(m);  rook_shift.append(s);  rook_table.append(t)

    for sq in range(64):
        m, s, t = find_magic(sq, BISH_MASKS[sq], rook=False)
        bish_magic.append(m);  bish_shift.append(s);  bish_table.append(t)

    return (rook_magic, rook_shift, rook_table,
            bish_magic, bish_shift, bish_table)

# Either regenerate or load from cache.
CACHE = Path(__file__).with_suffix('.pkl')
if CACHE.exists():
    (ROOK_MAGIC, ROOK_SHIFT, ROOK_TABLE,
     BISH_MAGIC, BISH_SHIFT, BISH_TABLE) = pickle.loads(CACHE.read_bytes())
else:
    (ROOK_MAGIC, ROOK_SHIFT, ROOK_TABLE,
     BISH_MAGIC, BISH_SHIFT, BISH_TABLE) = build_magic_tables()
    CACHE.write_bytes(pickle.dumps((
        ROOK_MAGIC, ROOK_SHIFT, ROOK_TABLE,
        BISH_MAGIC, BISH_SHIFT, BISH_TABLE
    )))
