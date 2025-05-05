

function undoMove() {
    try {
        fetch('http://127.0.0.1:5000/undo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            board.position(data.fen, true);
            game.load(data.fen);
            updateStatus();
        });
    } catch (err) {
        console.error('Lỗi khi giao tiếp với server:', err);
        alert('Không thể kết nối với server.');
    }
}