function highlightValidMoves(sourceSquare) {
    const moves = game.moves({ square: sourceSquare, verbose: true });

    // Xóa mọi highlight trước đó
    removeHighlights();

    // Nếu không có nước đi hợp lệ, dừng lại
    if (moves.length === 0) return;

    // Tô sáng các ô
    moves.forEach(move => {
        const square = move.to;
        const squareEl = document.querySelector(`[data-square="${square}"]`); // Tìm ô bằng data-square
        if (squareEl) {
            console.log(`Highlighting square: ${square}`); // Log các ô được tô sáng
            squareEl.classList.add('highlight');
        }
    });
}


function removeHighlights() {
    const highlightedSquares = document.querySelectorAll('.highlight'); // Chỉ tìm các ô có class highlight
    console.log('Squares to remove highlight from:', highlightedSquares); // Log các ô cần xóa
    highlightedSquares.forEach(square => {
        square.classList.remove('highlight'); // Xóa highlight
    });
}

function highlightPreviousMove(source, target) {
    const sourceSquareEl = document.querySelector(`[data-square="${source}"]`);
    const targetSquareEl = document.querySelector(`[data-square="${target}"]`);

    // Tô sáng với lớp khác thay vì 'highlight'
    if (sourceSquareEl) sourceSquareEl.classList.add('highlight-move');
    if (targetSquareEl) targetSquareEl.classList.add('highlight-move');
}

function removePreviousMoveHighlight() {
    const previousMoveSquares = document.querySelectorAll('.highlight-move');
    previousMoveSquares.forEach(square => {
        square.classList.remove('highlight-move');
    });
}