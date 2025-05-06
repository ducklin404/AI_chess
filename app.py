from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess.engine
import random

app = Flask(__name__)
CORS(app)

# Khởi tạo trò chơi cờ vua
board = chess.Board()

@app.route('/move', methods=['POST'])
def get_ai_move():
    global board

    # Lấy nước đi từ request của client
    data = request.json
    move = data.get('move')
    print(move)
    if len(move) > 1:
        try:
            # Áp dụng nước đi của người chơi
            board.push_uci(move)
        except ValueError:
            return jsonify({"error": "Invalid move"}), 400

    # Kiểm tra trạng thái ván cờ trước khi AI thực hiện nước đi
    if board.is_checkmate():
        print("Checkmate! Game over.")
        board.reset()  # Reset lại bàn cờ
        return jsonify({"message": "Checkmate! Game over. The game has been reset.", "fen": board.fen()})
    elif board.is_stalemate():
        print("Stalemate! Game over.")
        board.reset()  # Reset lại bàn cờ
        return jsonify({"message": "Stalemate! Game over. The game has been reset.", "fen": board.fen()})

    # Sử dụng AI để tính nước đi
    with chess.engine.SimpleEngine.popen_uci(r"C:\Users\ASUS\Downloads\stockfish-windows-x86-64-avx2(1)\stockfish\stockfish-windows-x86-64-avx2.exe") as engine:
        # Phân tích các nước đi với giới hạn thời gian 1 giây
        info = engine.analyse(board, chess.engine.Limit(time=1.0), multipv=5)  # multipv=5 để lấy top 5 nước đi

        # Lấy danh sách các nước đi
        possible_moves = [entry["pv"][0] for entry in info]  # Lấy nước đi đầu tiên trong mỗi dòng PV

        if not possible_moves:
            if board.is_checkmate():
                print("Checkmate! Game over.")
                board.reset()  # Reset lại bàn cờ
                return jsonify({"message": "Checkmate! Game over. The game has been reset.", "fen": board.fen()})
            elif board.is_stalemate():
                print("Stalemate! Game over.")
                board.reset()  # Reset lại bàn cờ
                return jsonify({"message": "Stalemate! Game over. The game has been reset.", "fen": board.fen()})
            else:
                print("No legal moves left!")
                board.reset()  # Reset lại bàn cờ
                return jsonify({"message": "No legal moves left! The game has been reset.", "fen": board.fen()})

        # Chọn ngẫu nhiên một nước đi
        ai_move = random.choice(possible_moves)

        # Thực hiện nước đi
        board.push(ai_move)

    # Kiểm tra trạng thái bàn cờ sau nước đi của AI
    if board.is_checkmate():
        print("Checkmate! Game over.")
        board.reset()  # Reset lại bàn cờ
        return jsonify({"message": "Checkmate! Game over. The game has been reset.", "fen": board.fen()})
    elif board.is_stalemate():
        print("Stalemate! Game over.")
        board.reset()  # Reset lại bàn cờ
        return jsonify({"message": "Stalemate! Game over. The game has been reset.", "fen": board.fen()})

    print(board.fen())
    return jsonify({"move": ai_move.uci(), "fen": board.fen()})

@app.route('/reset', methods=['POST'])
def reset_game():
    global board
    board.reset()
    print(board.fen())
    return jsonify({"fen": board.fen()})

@app.route('/undo', methods=['POST'])
def undo_move():
    global board
    board.pop()
    print(board.fen())
    return jsonify({"fen": board.fen()})

if __name__ == '__main__':
    app.run(debug=True)
