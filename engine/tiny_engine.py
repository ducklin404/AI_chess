from pre_calculate_moves import *
from copy import deepcopy                    
from nnue import NNUE, evaluate_position
import torch

INF = 10**9
popcnt = int.bit_count          
WK, WQ, BK, BQ = 1, 2, 4, 8

# Initialize NNUE model
nnue_model = NNUE()
# Load weights if available
try:
    nnue_model.load_state_dict(torch.load('nnue_weights.pth'))
    nnue_model.eval()
except:
    print("No NNUE weights found, using untrained model")

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
        if piece is None:  # Add safety check
            return

        # Store captured piece before making the move
        captured_piece = None
        if flag in ('x', 'ep'):
            cap_sq = to if flag == 'x' else (to - 8 if self.side else to + 8)
            captured_piece = self.piece_at(cap_sq)

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

        # Handle captures first (including en passant)
        if flag in ('x', 'ep'):
            cap_sq = to if flag == 'x' else (to - 8 if self.side else to + 8)
            cap_pc = self.piece_at(cap_sq)
            if cap_pc:
                # Remove captured piece
                self.bb[cap_pc] = self.bb[cap_pc] & ~(1 << cap_sq)

        # Handle special moves
        if flag == 'ck':          # Kingside castling
            if self.side:  # white
                # Move king
                self.bb['K'] = (self.bb['K'] & ~(1 << 4)) | (1 << 6)
                # Move rook
                self.bb['R'] = (self.bb['R'] & ~(1 << 7)) | (1 << 5)
            else:           # black
                # Move king
                self.bb['k'] = (self.bb['k'] & ~(1 << 60)) | (1 << 62)
                # Move rook
                self.bb['r'] = (self.bb['r'] & ~(1 << 63)) | (1 << 61)
        elif flag == 'cq':      # Queenside castling
            if self.side:
                # Move king
                self.bb['K'] = (self.bb['K'] & ~(1 << 4)) | (1 << 2)
                # Move rook
                self.bb['R'] = (self.bb['R'] & ~(1 << 0)) | (1 << 3)
            else:
                # Move king
                self.bb['k'] = (self.bb['k'] & ~(1 << 60)) | (1 << 58)
                # Move rook
                self.bb['r'] = (self.bb['r'] & ~(1 << 56)) | (1 << 59)
        else:
            # Handle pawn double push for en passant
            if piece in "Pp" and abs(to - frm) == 16:
                self.ep = (frm + to) // 2

            # Handle promotions
            if piece in "Pp" and to // 8 in (0, 7):
                promo = 'Q' if piece.isupper() else 'q'
                # Remove pawn from old position
                self.bb[piece] = self.bb[piece] & ~(1 << frm)
                # Add promoted piece to new position
                self.bb[promo] = self.bb[promo] | (1 << to)
            else:
                # Regular move
                # Remove piece from old position and add to new position
                self.bb[piece] = (self.bb[piece] & ~(1 << frm)) | (1 << to)

        # Update castling rights if rook is captured
        if flag == 'x':
            if to == 0:   self.castle &= ~WQ
            if to == 7:   self.castle &= ~WK
            if to == 56:  self.castle &= ~BQ
            if to == 63:  self.castle &= ~BK

        # Switch side to move
        self.side ^= 1
        self._refresh_occ()
        
        # Return captured piece for game state checking
        return captured_piece


    def piece_at(self, sq):
        """Get piece at square, with safety check"""
        if not 0 <= sq < 64:  # Add bounds check
            return None
        for p, b in self.bb.items():
            if b & (1 << sq): return p
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
def is_checkmate(board):
    """Check if the current position is checkmate"""
    # Check if king is in check
    ksq = lsb(board.bb['K' if board.side else 'k'])
    if not board.is_attacked(ksq, by_white=not board.side):
        return False
    
    # Check if any legal moves exist
    return not any(board.moves())

def is_stalemate(board):
    """Check if the current position is stalemate"""
    # Check if king is not in check
    ksq = lsb(board.bb['K' if board.side else 'k'])
    if board.is_attacked(ksq, by_white=not board.side):
        return False
    
    # Check if any legal moves exist
    return not any(board.moves())

def evaluate(board: Board):
    """Evaluate position using NNUE with bonus for king capture"""
    score = evaluate_position(board, nnue_model)
    
    # Add bonus for king capture
    if not board.bb['K']:  # White king is captured
        score = -10000  # Black wins
    elif not board.bb['k']:  # Black king is captured
        score = 10000   # White wins
    
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

def get_computer_move(board, depth=3):
    """Get the best move for the computer using negamax search"""
    score, best_move = negamax(board, depth, -INF, INF)
    return best_move

def play_computer_move(board):
    """Make the best move for the computer"""
    move = get_computer_move(board)
    if move:
        board.make(move)
        return move
    return None

def print_game_status(board):
    """Print the current game status"""
    # Check for king capture
    if not board.bb['K']:
        print("\nGame Over! Black wins by capturing the White king!")
        return True
    elif not board.bb['k']:
        print("\nGame Over! White wins by capturing the Black king!")
        return True

    # Check for checkmate
    if is_checkmate(board):
        winner = "Black" if board.side else "White"
        print(f"\nCheckmate! {winner} wins!")
        return True

    # Check for stalemate
    if is_stalemate(board):
        print("\nStalemate! Game is a draw.")
        return True

    # Check for insufficient material
    if is_insufficient_material(board):
        print("\nDraw by insufficient material!")
        return True

    # Check for threefold repetition
    if is_threefold_repetition(board):
        print("\nDraw by threefold repetition!")
        return True

    # Check for fifty-move rule
    if is_fifty_move_rule(board):
        print("\nDraw by fifty-move rule!")
        return True

    return False

def is_insufficient_material(board):
    """Check if there is insufficient material to checkmate"""
    # Count pieces
    piece_count = {p: popcnt(bb) for p, bb in board.bb.items()}
    
    # King vs King
    if sum(piece_count.values()) == 2:
        return True
        
    # King and Knight vs King
    if piece_count['N'] == 1 and piece_count['n'] == 0 and sum(piece_count.values()) == 3:
        return True
    if piece_count['n'] == 1 and piece_count['N'] == 0 and sum(piece_count.values()) == 3:
        return True
        
    # King and Bishop vs King
    if piece_count['B'] == 1 and piece_count['b'] == 0 and sum(piece_count.values()) == 3:
        return True
    if piece_count['b'] == 1 and piece_count['B'] == 0 and sum(piece_count.values()) == 3:
        return True
        
    return False

def is_threefold_repetition(board):
    """Check for threefold repetition"""
    # This is a simplified version - in a real implementation,
    # you would need to store the position history
    return False

def is_fifty_move_rule(board):
    """Check for fifty-move rule"""
    # This is a simplified version - in a real implementation,
    # you would need to track moves since last capture or pawn move
    return False

def print_board(board):
    """Print the current board state"""
    pieces = {0: '.', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
              7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'}
    
    print("\n  a b c d e f g h")
    print("  ---------------")
    for rank in range(7, -1, -1):
        print(f"{rank+1}|", end=" ")
        for file in range(8):
            sq = rank * 8 + file
            piece = board.piece_at(sq)
            if piece:
                print(piece, end=" ")
            else:
                print(".", end=" ")
        print(f"|{rank+1}")
    print("  ---------------")
    print("  a b c d e f g h\n")

def move_to_text(move):
    """Convert move to algebraic notation"""
    frm, to, flag = move
    move_text = f"{FILES[frm%8]}{frm//8+1}{FILES[to%8]}{to//8+1}"
    
    # Add special move indicators
    if flag == 'x':
        move_text += 'x'
    elif flag == 'ep':
        move_text += 'ep'
    elif flag == 'ck':
        move_text = 'O-O'
    elif flag == 'cq':
        move_text = 'O-O-O'
    
    return move_text

def print_move_history(moves):
    """Print move history in a readable format"""
    print("\nMove History:")
    print("-------------")
    for i, move in enumerate(moves):
        if i % 2 == 0:
            print(f"{i//2 + 1}.", end=" ")
        print(move_to_text(move), end=" ")
        if i % 2 == 1:
            print()
    print("\n-------------")

if __name__ == "__main__":
    # Example of playing against computer
    board = Board()
    print("Starting new game...")
    move_history = []
    
    while True:
        print_board(board)
        print_move_history(move_history)
        
        # Check game status
        if print_game_status(board):
            break
        
        # Player's move
        try:
            move_text = input("\nEnter your move (e.g., e2e4) or 'quit' to exit: ")
            if move_text.lower() == 'quit':
                break
                
            move = parse_move(board, move_text)
            captured_piece = board.make(move)
            move_history.append(move)
            
            # Check if king was captured
            if captured_piece in ['K', 'k']:
                print(f"\nGame Over! {'Black' if board.side else 'White'} wins by capturing the king!")
                break
            
            # Computer's move
            print("\nComputer is thinking...")
            comp_move = play_computer_move(board)
            if comp_move:
                move_history.append(comp_move)
                frm, to, flag = comp_move
                print(f"Computer plays: {move_to_text(comp_move)}")
                
                # Check if king was captured
                captured_piece = board.make(comp_move)
                if captured_piece in ['K', 'k']:
                    print(f"\nGame Over! {'Black' if board.side else 'White'} wins by capturing the king!")
                    break
            else:
                print("Game over!")
                break
                
        except Exception as e:
            print(f"Invalid move: {e}")
            continue
    