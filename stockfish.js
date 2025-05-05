async function makeMove(source, target) {
    
    let move = source + target; // Nước đi dạng UCI (vd: "e2e4")
    // Kiểm tra xem đây có phải là nước đi phong cấp không
    console.log(isPawnPromotion(piece, target));
    // Nếu là nước đi phong cấp, thêm hậu tố
    if (isPawnPromotion(piece, target)) {
        move += game.get(target).type ; // Thêm hậu tố
        console.log('Promotion move:', move);
    }

    else  console.log('Making move:', move);
  
    try {
        // Gửi yêu cầu POST đến server Python
        const response = await fetch('http://127.0.0.1:5000/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ move: move }),
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error); // Thông báo lỗi
            return 'snapback';
        }
        //bỏ highlight trước đó
        removePreviousMoveHighlight();
        AI_source = data.move.substring(0, 2);
        AI_target = data.move.substring(2, 4);
        highlightPreviousMove(AI_source, AI_target); // Tô sáng nước đi của AI
        
        // Cập nhật bàn cờ với nước đi của AI
        board.position(data.fen, true);
        game.load(data.fen);
        updateStatus();

    } catch (err) {
        console.error('Lỗi khi giao tiếp với server:', err);
        //alert('Không thể kết nối với server.');
        return 'snapback';
    }
}


function resetGame() {
    try {
        fetch('http://127.0.0.1:5000/reset', {
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

function isPawnPromotion(piece, target) {
    const promotionRanks = ['1', '8']; // Các hàng để phong cấp
    return piece.toLowerCase() === 'p' && promotionRanks.includes(target[1]);
}


function shouldPromotePawn(game, source, target) {
    const piece = game.get(source);
    const isPawn = piece && piece.type === 'p'; // Quân tốt
    const isPromotingRank = target[1] === '8' || target[1] === '1'; // Hàng cuối
    return isPawn && isPromotingRank;
}

