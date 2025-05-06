from pre_calculate_moves import *
from copy import deepcopy                    

INF = 10**9
popcnt = int.bit_count          
WK, WQ, BK, BQ = 1, 2, 4, 8

class Board:

    def __init__(self,
                 fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.bb = {c: 0 for c in "PNBRQKpnbrqk"}
        self.side = 1        # 1 = white to move
        self.ep = None
        self.castle = WK | WQ | BK | BQ
        self._set_fen(fen)

    # FEN to bitboards 
    def _set_fen(self, fen: str):
        place, side, castles, ep_sq, *_ = fen.split()

        # pieces
        rank, file = 7, 0
        for ch in place:
            if ch == "/":
                rank -= 1; file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                self.bb[ch] |= 1 << (rank * 8 + file)
                file += 1

        # side to move
        self.side = 1 if side == "w" else 0

        # ─── CASTLING
        self.castle = 0
        if "K" in castles: self.castle |= WK
        if "Q" in castles: self.castle |= WQ
        if "k" in castles: self.castle |= BK
        if "q" in castles: self.castle |= BQ

        # EP square
        self.ep = None if ep_sq == "-" else SQUARES[ep_sq]   # SQUARES helper given below
        self._refresh_occ()

    # helper to rebuild occ
    def _refresh_occ(self):
        self.white_occ = sum(self.bb[p] for p in "PNBRQK") & 0xFFFF_FFFF_FFFF_FFFF
        self.black_occ = sum(self.bb[p] for p in "pnbrqk") & 0xFFFF_FFFF_FFFF_FFFF
        
    def is_attacked(self, sq: int, by_white: bool) -> bool:
        """Return True if `sq` is attacked by the given side."""
        occ = self.white_occ | self.black_occ

        if by_white:
            if PAWN_ATTACKS_B[sq] & self.bb['P']: return True
            if KNIGHT_ATTACKS[sq] & self.bb['N']: return True
            if KING_ATTACKS[sq]   & self.bb['K']: return True
            if bishop_attacks(sq, occ) & (self.bb['B'] | self.bb['Q']): return True
            if rook_attacks(sq, occ)   & (self.bb['R'] | self.bb['Q']): return True
        else:  # by black
            if PAWN_ATTACKS_W[sq] & self.bb['p']: return True
            if KNIGHT_ATTACKS[sq] & self.bb['n']: return True
            if KING_ATTACKS[sq]   & self.bb['k']: return True
            if bishop_attacks(sq, occ) & (self.bb['b'] | self.bb['q']): return True
            if rook_attacks(sq, occ)   & (self.bb['r'] | self.bb['q']): return True
        return False
    
    

    # ────────────────────────────────────────────────────────────────────
    # 2.  Move generation  
    # ────────────────────────────────────────────────────────────────────
    def moves(self):
        occ   = self.white_occ | self.black_occ
        empty = ~occ & 0xFFFF_FFFF_FFFF_FFFF
        if self.side:                         # WHITE 
            own, enemy = self.white_occ, self.black_occ
            pawn_bb    = self.bb['P']
            step, dbl, seventh, promo_rank = 8, 16, 6, 7
            pawn_att   = PAWN_ATTACKS_W
        else:                                 # BLACK
            own, enemy = self.black_occ, self.white_occ
            pawn_bb    = self.bb['p']
            step, dbl, seventh, promo_rank = -8, -16, 1, 0
            pawn_att   = PAWN_ATTACKS_B
            
            
        # King
        ksq = lsb(self.bb['K' if self.side else 'k'])
        targets = KING_ATTACKS[ksq]
        for to in squares(targets & empty):     yield ksq, to, ''
        for to in squares(targets & enemy):     yield ksq, to, 'x'

        # CASTLING moves 
        if self.side:   # WHITE
            if self.castle & WK:                    
                if not (occ & 0x60) and \
                   not self.is_attacked(4, False) and \
                   not self.is_attacked(5, False) and \
                   not self.is_attacked(6, False):
                    yield 4, 6, 'ck'                   
            if self.castle & WQ:                        
                if not (occ & 0x0E) and \
                   not self.is_attacked(4, False) and \
                   not self.is_attacked(3, False) and \
                   not self.is_attacked(2, False):
                    yield 4, 2, 'cq'                 
        else:         # BLACK
            if self.castle & BK:
                if not (occ & 0x6000000000000000) and \
                   not self.is_attacked(60, True) and \
                   not self.is_attacked(61, True) and \
                   not self.is_attacked(62, True):
                    yield 60, 62, 'ck'                
            if self.castle & BQ:
                if not (occ & 0x0E00000000000000) and \
                   not self.is_attacked(60, True) and \
                   not self.is_attacked(59, True) and \
                   not self.is_attacked(58, True):
                    yield 60, 58, 'cq'                  

        # Knights
        bb = self.bb['N' if self.side else 'n']
        while bb:
            frm, bb = pop_lsb(bb)
            targets = KNIGHT_ATTACKS[frm]
            for to in squares(targets & empty):  yield frm, to, ''
            for to in squares(targets & enemy):  yield frm, to, 'x'

        # Bishops
        bb = self.bb['B' if self.side else 'b']
        while bb:
            frm, bb = pop_lsb(bb)
            targets = bishop_attacks(frm, occ)
            for to in squares(targets & empty):  yield frm, to, ''
            for to in squares(targets & enemy):  yield frm, to, 'x'

        # Rooks
        bb = self.bb['R' if self.side else 'r']
        while bb:
            frm, bb = pop_lsb(bb)
            targets = rook_attacks(frm, occ)
            for to in squares(targets & empty):  yield frm, to, ''
            for to in squares(targets & enemy):  yield frm, to, 'x'

        # Queen
        bb = self.bb['Q' if self.side else 'q']
        while bb:
            frm, bb = pop_lsb(bb)
            targets = queen_attacks(frm, occ)
            for to in squares(targets & empty):  yield frm, to, ''
            for to in squares(targets & enemy):  yield frm, to, 'x'

        # King (no castling for brevity)
        ksq = lsb(self.bb['K' if self.side else 'k'])
        targets = KING_ATTACKS[ksq]
        for to in squares(targets & empty):     yield ksq, to, ''
        for to in squares(targets & enemy):     yield ksq, to, 'x'

        # Pawns – pushes
        push = (pawn_bb << step) & empty if self.side else (pawn_bb >> -step) & empty
        for to in squares(push):
            frm = to - step
            yield frm, to, ''                        

            # double-push
            start_rank = 1 if self.side else 6
            ahead = to + step       
            if frm // 8 == start_rank and (1 << ahead) & empty:
                yield frm, ahead, ''              

        # Pawn captures (+ en-passant)
        bb = pawn_bb
        while bb:
            frm, bb = pop_lsb(bb)
            caps = pawn_att[frm] & enemy
            for to in squares(caps):           yield frm, to, 'x'
            # en-passant
            if self.ep is not None and pawn_att[frm] & (1 << self.ep):
                yield frm, self.ep, 'ep'

    # ────────────────────────────────────────────────────────────────────
    # 3.  Make / unmake  
    # ────────────────────────────────────────────────────────────────────
    def make(self, move):
        frm, to, flag = move
        piece = self.piece_at(frm)

        # clear previous EP
        self.ep = None

        # CASTLING
        if piece == 'K':
            self.castle &= ~(WK | WQ)         
        elif piece == 'k':
            self.castle &= ~(BK | BQ)
        elif piece == 'R':
            if frm == 0:  self.castle &= ~WQ  
            if frm == 7:  self.castle &= ~WK 
        elif piece == 'r':
            if frm == 56: self.castle &= ~BQ  
            if frm == 63: self.castle &= ~BK 

        if flag == 'ck':          
            if self.side:  # white
                self.bb['K'] ^= (1<<4) | (1<<6)   
                self.bb['R'] ^= (1<<7) | (1<<5)  
            else:           # black
                self.bb['k'] ^= (1<<60) | (1<<62)
                self.bb['r'] ^= (1<<63) | (1<<61)
        elif flag == 'cq':      
            if self.side:
                self.bb['K'] ^= (1<<4) | (1<<2)   
                self.bb['R'] ^= (1<<0) | (1<<3)   
            else:
                self.bb['k'] ^= (1<<60) | (1<<58)
                self.bb['r'] ^= (1<<56) | (1<<59)
        else:
            # pawn double → EP
            if piece in "Pp" and abs(to - frm) == 16:
                self.ep = (frm + to) // 2

            # promotions
            if piece in "Pp" and to // 8 in (0, 7):
                promo = 'Q' if piece.isupper() else 'q'
                self.bb[piece] ^= 1 << frm
                self.bb[promo] |= 1 << to
            else:
                self.bb[piece] ^= (1 << frm) | (1 << to)

            # captures (incl. EP)
            if flag in ('x', 'ep'):
                cap_sq = to if flag == 'x' else (to - 8 if self.side else to + 8)
                cap_pc = self.piece_at(cap_sq)
                if cap_pc:
                    self.bb[cap_pc] ^= 1 << cap_sq

        if flag == 'x':
            if to == 0:   self.castle &= ~WQ
            if to == 7:   self.castle &= ~WK
            if to == 56:  self.castle &= ~BQ
            if to == 63:  self.castle &= ~BK

        # side to move
        self.side ^= 1
        self._refresh_occ()


    def piece_at(self, sq):
        for p,b in self.bb.items():
            if b >> sq & 1: return p
        return None

# ────────────────────────────────────────────────────────────────────────────
# 4.  helpers  (bit tricks)
# ────────────────────────────────────────────────────────────────────────────
def lsb(b): return (b & -b).bit_length()-1
def pop_lsb(b):
    l = b & -b
    return l.bit_length()-1, b ^ l
def squares(bb):
    while bb:
        s, bb = pop_lsb(bb)
        yield s
        
FILES = "abcdefgh"
SQUARES = {f"{FILES[f]}{r+1}": r*8+f for r in range(8) for f in range(8)}
def coord_to_sq(text: str) -> int:

    f, r = text[0], text[1]
    return (int(r) - 1) * 8 + (ord(f) - ord("a"))



def parse_move(board: Board, text: str):
    t = text.strip().lower().replace("-", "")
    if t in ("oo", "o0", "0o", "00"):          # O-O
        return (4, 6, "ck") if board.side else (60, 62, "ck")
    if t in ("ooo", "o00", "0oo", "000"):      # O-O-O
        return (4, 2, "cq") if board.side else (60, 58, "cq")

    if len(t) < 4:
        raise ValueError("Not enough characters")

    frm = coord_to_sq(t[0:2])
    to  = coord_to_sq(t[2:4])

    # promotion piece ignored – engine always makes a queen
    flag = ""
    if t.endswith("ep"):
        flag = "ep"
    else:
        target_occ = board.black_occ if board.side else board.white_occ
        if (1 << to) & target_occ:
            flag = "x"
    return frm, to, flag

def legal_moves(board: Board):
    for mv in board.moves():
        child = Board()          
        child.__dict__ = board.__dict__.copy()  
        child.make(mv)
        # king square after move
        ksq = lsb(child.bb['K' if child.side else 'k'])
        if not child.is_attacked(ksq, by_white=not child.side):
            yield mv

# ────────────────────────────────────────────────────────────────────────────
# 5.  Negamax + αβ
# ────────────────────────────────────────────────────────────────────────────
def evaluate(board: Board):
    """Material only (P=100, N/B=300, R=500, Q=900)."""
    piece_val = dict(P=100,N=300,B=300,R=500,Q=900,
                     p=-100,n=-300,b=-300,r=-500,q=-900)
    score = sum(piece_val.get(p,0) * popcnt(bb)
                for p,bb in board.bb.items())
    return score if board.side else -score

def negamax(board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board), None
    best_move = None
    for mv in board.moves():
        child = deepcopy(board)
        child.make(mv)
        sc, _ = negamax(child, depth-1, -beta, -alpha)
        sc = -sc
        if sc > alpha:
            alpha, best_move = sc, mv
            if alpha >= beta: break           # β-cutoff
    return alpha, best_move


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for mv in board.moves():
        child = deepcopy(board)
        child.make(mv)
        nodes += perft(child, depth - 1)
    return nodes


# ────────────────────────────────────────────────────────────────────────────
# 6.  Quick demo
# ────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     b = Board()                               # start position
#     score, best = negamax(b, depth=3, alpha=-INF, beta=INF)
#     print("depth-3 score:", score)
#     print("best move   :", best)               # (from, to, flag)


if __name__ == "__main__":
    b = Board()
    for d in range(1, 6):
        print(f"perft({d}) =", perft(b, d))
    