# -*- coding: utf-8 -*-

import random
from PyQt5.QtWidgets import (QApplication, QFrame, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QLabel, QHBoxLayout, QGridLayout)
from PyQt5.QtCore import QPoint, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPalette, QPen, QBrush, QPolygon
import sys
import ollama
import json
from enum import Enum

import re

class PieceType(Enum):
    EMPTY = 0
    BLACK = 1
    BLACK_KING = 2
    RED = 3
    RED_KING = 4
    
class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        
        # Title label
        self.title = QLabel("Qwen2.5:Checkers")
        self.title.setStyleSheet("""
            color: #FFFFFF;
            font-size: 14px;
            font-weight: bold;
            padding-left: 10px;
        """)
        
        # Buttons
        self.minimize_button = QPushButton("−")
        self.close_button = QPushButton("×")
        
        # Button styling
        button_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                color: #FFFFFF;
                font-size: 16px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """
        self.minimize_button.setStyleSheet(button_style)
        self.close_button.setStyleSheet(button_style)
        
        # Layout
        self.layout.addWidget(self.title)
        self.layout.addStretch()
        self.layout.addWidget(self.minimize_button)
        self.layout.addWidget(self.close_button)
        
        # Connect buttons
        self.minimize_button.clicked.connect(self.parent.showMinimized)
        self.close_button.clicked.connect(self.parent.close)
        
        # Window dragging
        self.start = QPoint(0, 0)
        self.pressing = False

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            end = self.mapToGlobal(event.pos())
            movement = end - self.start
            self.parent.setGeometry(
                self.parent.x() + movement.x(),
                self.parent.y() + movement.y(),
                self.parent.width(),
                self.parent.height()
            )
            self.start = end

    def mouseReleaseEvent(self, event):
        self.pressing = False

class CheckerSquare(QPushButton):
    def __init__(self, row, col, piece=PieceType.EMPTY):
        super().__init__()
        self.row = row
        self.col = col
        self.piece = piece
        self.selected = False
        self.possible_move = False
        # Perfect square size
        self.setFixedSize(60, 60)
        # Remove ALL padding/margins that could distort the grid
        self.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet(self._get_style())

    def _get_style(self):
        bg_color = "#4A3B22" if (self.row + self.col) % 2 == 0 else "#2A2016"
        border = "2px solid #FFD700" if self.selected else "none"
        highlight = "background-color: #445C24;" if self.possible_move else ""
        
        return f"""
            QPushButton {{
                background-color: {bg_color};
                border: {border};
                {highlight}
                margin: 0px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {'#5A4B32' if (self.row + self.col) % 2 == 0 else '#3A3026'};
            }}
        """

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.piece != PieceType.EMPTY:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw piece with darker theme colors
            color = QColor("#FF6666") if self.piece in [PieceType.RED, PieceType.RED_KING] else QColor("#333333")
            painter.setPen(QPen(color.darker(), 2))
            painter.setBrush(QBrush(color))
            
            center = self.rect().center()
            radius = min(self.width(), self.height()) * 0.35
            painter.drawEllipse(center, radius, radius)
            
            if self.piece in [PieceType.BLACK_KING, PieceType.RED_KING]:
                crown_color = QColor("#FFD700")
                painter.setPen(QPen(crown_color, 2))
                painter.setBrush(QBrush(crown_color))
                
                crown_size = radius * 0.6
                points = [
                    QPoint(int(center.x() - crown_size), int(center.y() - crown_size * 0.5)),
                    QPoint(int(center.x() + crown_size), int(center.y() - crown_size * 0.5)),
                    QPoint(int(center.x()), int(center.y() + crown_size * 0.5))
                ]
                painter.drawPolygon(QPolygon(points))

class GameState:
    def __init__(self):
        self.board = [[PieceType.EMPTY] * 8 for _ in range(8)]
        self.current_player = PieceType.RED
        self.move_history = []  # Add move history tracking
        self.initialize_board()
    
    def initialize_board(self):
        """Initialize the standard checkers starting position"""
        # Place black pieces (rows 0-2)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = PieceType.BLACK
        
        # Place red pieces (rows 5-7)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = PieceType.RED

    def _is_valid_position(self, row, col):
        """Check if position is within board bounds"""
        return 0 <= row < 8 and 0 <= col < 8

    def _is_opponent_piece(self, row, col, current_piece):
        """Check if piece at position belongs to opponent"""
        if not self._is_valid_position(row, col):
            return False
        target = self.board[row][col]
        return ((current_piece in [PieceType.RED, PieceType.RED_KING] and 
                target in [PieceType.BLACK, PieceType.BLACK_KING]) or
                (current_piece in [PieceType.BLACK, PieceType.BLACK_KING] and 
                target in [PieceType.RED, PieceType.RED_KING]))

    def _get_capture_moves(self, row, col):
        """Get all possible capture moves for a piece including multi-jumps"""
        piece = self.board[row][col]
        if piece == PieceType.EMPTY:
            return []
            
        moves = []
        directions = []
        
        # Kings can move in all directions
        if piece in [PieceType.RED_KING, PieceType.BLACK_KING]:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Regular pieces can only move forward
            if piece == PieceType.RED:
                directions = [(-1, -1), (-1, 1)]  # Red moves up
            else:  # BLACK
                directions = [(1, -1), (1, 1)]    # Black moves down
            
        # Check each direction for captures
        for dr, dc in directions:
            new_row, new_col = row + 2*dr, col + 2*dc
            jump_row, jump_col = row + dr, col + dc
            
            if (self._is_valid_position(new_row, new_col) and 
                self.board[new_row][new_col] == PieceType.EMPTY and
                self._is_opponent_piece(jump_row, jump_col, piece)):
                moves.append((new_row, new_col))
                
        return moves

    def _get_all_captures(self):
        """Get all pieces that have capture moves available"""
        captures = set()
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if ((self.current_player == PieceType.RED and piece in [PieceType.RED, PieceType.RED_KING]) or
                    (self.current_player == PieceType.BLACK and piece in [PieceType.BLACK, PieceType.BLACK_KING])):
                    moves = self._get_capture_moves(row, col)
                    if moves:
                        captures.add((row, col))
        return captures

    def get_valid_moves(self, row, col):
        """Get all valid moves for a piece"""
        piece = self.board[row][col]
        if piece == PieceType.EMPTY:
            return []
            
        # First check for captures as they are mandatory
        captures = self._get_capture_moves(row, col)
        if captures:
            return captures
            
        moves = []
        directions = []
        
        # Define movement directions based on piece type
        if piece in [PieceType.RED_KING, PieceType.BLACK_KING]:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            if piece == PieceType.RED:
                directions = [(-1, -1), (-1, 1)]  # Red moves up
            else:  # BLACK
                directions = [(1, -1), (1, 1)]    # Black moves down
            
        # Check regular moves
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col) and self.board[new_row][new_col] == PieceType.EMPTY:
                moves.append((new_row, new_col))
                
        return moves

    def is_valid_move(self, from_pos, to_pos):
        """Validate a move"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        if not self._is_valid_position(from_row, from_col) or not self._is_valid_position(to_row, to_col):
            return False
            
        piece = self.board[from_row][from_col]
        
        # Verify correct player's turn
        if self.current_player == PieceType.RED:
            if piece not in [PieceType.RED, PieceType.RED_KING]:
                return False
        else:
            if piece not in [PieceType.BLACK, PieceType.BLACK_KING]:
                return False
                
        # Get valid moves and check if the target position is valid
        valid_moves = self.get_valid_moves(from_row, from_col)
        return (to_row, to_col) in valid_moves

    def make_move(self, from_pos, to_pos):
        """Execute a move with validation and history tracking"""
        if not self.is_valid_move(from_pos, to_pos):
            raise ValueError("Invalid move")
            
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.board[from_row][from_col]
        
        # Record move before executing it
        move_record = {
            'from': from_pos,
            'to': to_pos,
            'piece': piece,
            'captured': None
        }
        
        # Move piece
        self.board[from_row][from_col] = PieceType.EMPTY
        
        # Handle king promotion for both colors
        if piece == PieceType.RED and to_row == 0:
            piece = PieceType.RED_KING
        elif piece == PieceType.BLACK and to_row == 7:
            piece = PieceType.BLACK_KING
            
        self.board[to_row][to_col] = piece
        
        # Handle captures
        if abs(to_row - from_row) == 2:
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            move_record['captured'] = self.board[captured_row][captured_col]  # Record captured piece
            self.board[captured_row][captured_col] = PieceType.EMPTY
            
            # Check for additional captures
            additional_captures = self._get_capture_moves(to_row, to_col)
            if not additional_captures:
                self.current_player = PieceType.BLACK if self.current_player == PieceType.RED else PieceType.RED
        else:
            # Switch turns for non-capture moves
            self.current_player = PieceType.BLACK if self.current_player == PieceType.RED else PieceType.RED
        
        self.move_history.append(move_record)

    def get_board_state(self):
        """Get string representation of board state for AI"""
        state = []
        for row in range(8):
            row_state = []
            for col in range(8):
                piece = self.board[row][col]
                if piece == PieceType.EMPTY:
                    row_state.append('.')
                elif piece == PieceType.BLACK:
                    row_state.append('b')
                elif piece == PieceType.BLACK_KING:
                    row_state.append('B')
                elif piece == PieceType.RED:
                    row_state.append('r')
                elif piece == PieceType.RED_KING:
                    row_state.append('R')
            state.append(''.join(row_state))
        return '\n'.join(state)

class AIWorkerThread(QThread):
    move_ready = pyqtSignal(tuple)
    error = pyqtSignal(str)
    
    def __init__(self, board_state):
        super().__init__()
        self.board_state = board_state
        self.retry_count = 0
        self.max_retries = 3  # Limit retries to prevent infinite loops

    def get_moves(self):
        """Get all valid moves for black pieces"""
        board = [list(row) for row in self.board_state.split('\n')]
        moves = []
        
        # First check for captures
        for row in range(8):
            for col in range(8):
                if board[row][col] in ['b', 'B']:  # Regular or king black piece
                    # Check possible directions based on piece type
                    directions = [(1, 1), (1, -1)]  # Forward for regular pieces
                    if board[row][col] == 'B':  # King can move all directions
                        directions += [(-1, 1), (-1, -1)]  # Add backward moves for kings
                        
                    for dr, dc in directions:
                        jump_row = row + 2*dr
                        jump_col = col + 2*dc
                        mid_row = row + dr
                        mid_col = col + dc
                        
                        if (0 <= jump_row < 8 and 0 <= jump_col < 8 and
                            0 <= mid_row < 8 and 0 <= mid_col < 8 and
                            board[jump_row][jump_col] == '.' and
                            board[mid_row][mid_col] in ['r', 'R']):
                            moves.append(((row, col), (jump_row, jump_col)))
                            
        # If no captures, get regular moves
        if not moves:
            for row in range(8):
                for col in range(8):
                    if board[row][col] in ['b', 'B']:
                        directions = [(1, 1), (1, -1)]  # Forward for regular pieces
                        if board[row][col] == 'B':  # King can move all directions
                            directions += [(-1, 1), (-1, -1)]  # Add backward moves for kings
                            
                        for dr, dc in directions:
                            new_row = row + dr
                            new_col = col + dc
                            if (0 <= new_row < 8 and 0 <= new_col < 8 and
                                board[new_row][new_col] == '.'):
                                moves.append(((row, col), (new_row, new_col)))
        
        return moves

    def parse_move_from_response(self, content):
        """Parse move from AI response with better error handling"""
        try:
            # Look for move coordinates in the response
            from_pattern = r'<from>(\d+),(\d+)</from>'
            to_pattern = r'<to>(\d+),(\d+)</to>'
            
            from_match = re.search(from_pattern, content)
            to_match = re.search(to_pattern, content)
            
            if not from_match or not to_match:
                return None
                
            from_pos = (int(from_match.group(1)), int(from_match.group(2)))
            to_pos = (int(to_match.group(1)), int(to_match.group(2)))
            
            return from_pos, to_pos
            
        except Exception as e:
            print(f"Move parsing error: {str(e)}")
            return None

    def select_random_move(self, moves):
        """Select a random valid move as fallback"""
        if moves:
            return random.choice(moves)
        return None



    def run(self):
        try:
            moves = self.get_moves()
            if not moves:
                self.error.emit("No valid moves available")
                return

            # Create a detailed analysis of each potential move
            moves_str = []
            for (start_row, start_col), (end_row, end_col) in moves:
                becomes_king = end_row == 7
                is_capture = abs(end_row - start_row) == 2
                
                move_str = (
                    f"From position ({start_row},{start_col}) to ({end_row},{end_col})\n"
                    f"  - Starting square: {'edge' if start_col in [0,7] else 'interior'} of row {start_row}\n"
                    f"  - Ending square: {'edge' if end_col in [0,7] else 'interior'} of row {end_row}\n"
                    f"  - Movement: {'diagonal capture' if is_capture else 'diagonal move'} "
                    f"{'towards king row' if end_row > start_row else 'away from king row'}"
                )
                if becomes_king:
                    move_str += "\n  - Results in king promotion!"
                if is_capture:
                    captured_row = (start_row + end_row) // 2
                    captured_col = (start_col + end_col) // 2
                    move_str += f"\n  - Captures opponent's piece at ({captured_row},{captured_col})"
                moves_str.append(move_str)

            # Create a visual representation of the board with coordinates
            board_rows = self.board_state.split('\n')
            board_display = []
            for i, row in enumerate(board_rows):
                board_display.append(f"Row {i:<2} {row}  {'(top)' if i == 0 else '(bottom)' if i == 7 else ''}")

            prompt = f"""You are an expert checkers AI analyzing this position carefully. Here's the complete game state:

            BOARD VISUALIZATION (8x8 grid):
            {chr(10).join(board_display)}
            Col:     0 1 2 3 4 5 6 7

            PIECE NOTATION AND MOVEMENT:
            . = Empty square
            b = Your black piece (moves downward toward row 7)
            B = Your black king (moves in any diagonal direction)
            r = Red piece (moves upward toward row 0)
            R = Red king (moves in any diagonal direction)

            BOARD GEOMETRY:
            - Center squares: (3,3), (3,4), (4,3), (4,4)
            - Edge columns: 0 and 7 (limited diagonal movement)
            - Your king row: Row 7 (bottom)
            - Opponent's king row: Row 0 (top)
            - Diagonal movement follows pattern:
              Regular moves: row +/- 1, col +/- 1
              Captures: row +/- 2, col +/- 2

            MOVEMENT RULES:
            1. Regular pieces move diagonally forward only
               - Your black pieces move toward higher row numbers
               - Opponent's red pieces move toward lower row numbers
            2. Kings move diagonally in any direction (your regular piece becomes a king when reaching the players wall-side)
            3. Captures are mandatory when available
            4. Multiple captures must be completed in the same turn
            5. Pieces become kings when reaching the opposite end
            6. Edge pieces have only one diagonal direction available

            AVAILABLE MOVES (with spatial analysis):
            {chr(10).join(moves_str)}

            POSITION EVALUATION CRITERIA:
            1. Material Count
               - Each regular piece = 1 point
               - Each king = 2 points
               - Captured pieces shift material balance
            2. Spatial Control
               - Center squares provide maximum mobility
               - Connected pieces support each other
               - Back rank control prevents enemy kings
               - Edge pieces have limited tactical value
            3. King Creation
               - Proximity to king row (row 7 for black)
               - Protected paths to promotion
               - King creation opportunities
            4. Tactical Elements
               - Available captures
               - Multiple jump sequences
               - Piece protection and exposure
               - Trapped opponent pieces

            Choose the strongest move based on these factors. Respond with ONLY:
            <move>
            <from>row,col</from>
            <to>row,col</to>
            </move>"""

            response = ollama.chat(model='qwen2.5:14b', messages=[
                {
                    'role': 'system', 
                    'content': 'You are a checkers grandmaster AI. You understand spatial relationships, piece coordination, and the strategic implications of every move on the 8x8 board.'
                },
                {'role': 'user', 'content': prompt}
            ])
            
            content = response['message']['content']
            move = self.parse_move_from_response(content)
            
            if move and move in moves:
                self.move_ready.emit(move)
            else:
                # Fallback to simpler prompt if sophisticated analysis fails
                simplified_prompt = f"""You are playing checkers as black. Here's a simple board view:
                {self.board_state}

                Choose the best move from:
                {', '.join(f'({start[0]},{start[1]}) to ({end[0]},{end[1]})' for start, end in moves)}

                Respond with ONLY:
                <move>
                <from>row,col</from>
                <to>row,col</to>
                </move>"""
                
                response = ollama.chat(model='qwen2.5:14b', messages=[
                    {'role': 'system', 'content': 'You are playing checkers as black.'},
                    {'role': 'user', 'content': simplified_prompt}
                ])
                
                content = response['message']['content']
                move = self.parse_move_from_response(content)
                
                if move and move in moves:
                    self.move_ready.emit(move)
                else:
                    # Last resort: random valid move
                    fallback_move = self.select_random_move(moves)
                    if fallback_move:
                        print("Using fallback move due to invalid AI response")
                        self.move_ready.emit(fallback_move)
                    else:
                        self.error.emit("No valid moves available")
                        
        except Exception as e:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                print(f"AI attempt {self.retry_count} failed, retrying...")
                self.run()
            else:
                print(f"AI failed after {self.max_retries} attempts")
                self.error.emit(f"AI failed after {self.max_retries} attempts")
            
class CheckersGame(QMainWindow):
    def __init__(self):
        super().__init__(flags=Qt.Window | Qt.FramelessWindowHint)  # This is the key change
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.game_state = GameState()
        self.selected_square = None
        self.possible_moves = []
        self.game_over = False
        self.board_locked = False
        self.setup_ui()

    def setup_ui(self):
        """Initialize the game's user interface with dark theme"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create outer frame for window border
        outer_frame = QFrame()
        outer_frame.setObjectName("outerFrame")
        outer_frame.setStyleSheet("""
            QFrame#outerFrame {
                background-color: #1A1A1A;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)
        
        outer_layout = QVBoxLayout(outer_frame)
        outer_layout.setContentsMargins(0, 0, 0, 10)
        outer_layout.setSpacing(0)
        
        # Add custom title bar
        self.title_bar = TitleBar(self)
        outer_layout.addWidget(self.title_bar)
        

        # Create the game board
        board_widget = QWidget()
        board_layout = QGridLayout(board_widget)
        # Remove ALL spacing in the grid
        board_layout.setSpacing(0)
        board_layout.setContentsMargins(0, 0, 0, 0)

        self.squares = []
        for row in range(8):
            row_squares = []
            for col in range(8):
                square = CheckerSquare(row, col, self.game_state.board[row][col])
                square.clicked.connect(lambda checked, r=row, c=col: self.square_clicked(r, c))
                board_layout.addWidget(square, row, col)
                row_squares.append(square)
            self.squares.append(row_squares)
            
        outer_layout.addWidget(board_widget, alignment=Qt.AlignCenter)
        
        # Create a fixed-height container for the status label
        status_container = QWidget()
        status_container.setFixedHeight(50)  # Fixed height prevents layout shifts
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        # Status label with consistent styling
        self.status_label = QLabel("Your turn (Red)")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #FF6666;
                font-size: 16px;
                font-weight: bold;
                background-color: transparent;
                padding: 10px;
                min-height: 30px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        outer_layout.addWidget(status_container)
        layout.addWidget(outer_frame)
        
        # Set window size
        self.setFixedSize(500, 580)
        
    def update_status(self, text, color="#FF6666"):
        """Update status label with consistent styling"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 16px;
                font-weight: bold;
                background-color: transparent;
                padding: 10px;
                min-height: 30px;
            }}
        """)


    def lock_board(self):
        """Lock the board to prevent user interaction"""
        self.board_locked = True
        for row in self.squares:
            for square in row:
                square.setEnabled(False)

    def unlock_board(self):
        """Unlock the board to allow user interaction"""
        self.board_locked = False
        for row in self.squares:
            for square in row:
                square.setEnabled(True)

    def update_board_ui(self):
        """Update the visual state of the board"""
        for row in range(8):
            for col in range(8):
                square = self.squares[row][col]
                square.piece = self.game_state.board[row][col]
                square.selected = (row, col) == self.selected_square
                square.possible_move = (row, col) in self.possible_moves
                square.setStyleSheet(square._get_style())
                square.update()

    def check_winner(self):
        """Check for winner and update game state accordingly"""
        black_pieces = red_pieces = 0
        for row in range(8):
            for col in range(8):
                piece = self.game_state.board[row][col]
                if piece in [PieceType.BLACK, PieceType.BLACK_KING]:
                    black_pieces += 1
                elif piece in [PieceType.RED, PieceType.RED_KING]:
                    red_pieces += 1
                    
        if black_pieces == 0:
            self.game_over = True
            self.status_label.setText("Game Over - Red Wins!")
            self.status_label.setStyleSheet("color: #FF4444; font-size: 20px; font-weight: bold;")
            return True
        elif red_pieces == 0:
            self.game_over = True
            self.status_label.setText("Game Over - Black Wins!")
            self.status_label.setStyleSheet("color: #1A1A1A; font-size: 20px; font-weight: bold;")
            return True
            
        # Check if current player has any valid moves
        has_moves = False
        for row in range(8):
            for col in range(8):
                piece = self.game_state.board[row][col]
                if ((self.game_state.current_player == PieceType.RED and 
                     piece in [PieceType.RED, PieceType.RED_KING]) or
                    (self.game_state.current_player == PieceType.BLACK and 
                     piece in [PieceType.BLACK, PieceType.BLACK_KING])):
                    if self.game_state.get_valid_moves(row, col):
                        has_moves = True
                        break
            if has_moves:
                break
                
        if not has_moves:
            self.game_over = True
            winner = "Black" if self.game_state.current_player == PieceType.RED else "Red"
            self.status_label.setText(f"Game Over - {winner} Wins!")
            self.status_label.setStyleSheet(
                f"color: {'#1A1A1A' if winner == 'Black' else '#FF4444'}; "
                "font-size: 20px; font-weight: bold;"
            )
            return True
            
        return False

    def square_clicked(self, row, col):
        """Handle square click events"""
        if self.game_over or self.board_locked:
            return  # Prevent moves when game is over or board is locked
            
        if self.game_state.current_player == PieceType.BLACK:
            return  # Not player's turn
            
        piece = self.game_state.board[row][col]
        
        # If no piece is selected and clicked on own piece
        if not self.selected_square and piece in [PieceType.RED, PieceType.RED_KING]:
            self.selected_square = (row, col)
            self.possible_moves = self.game_state.get_valid_moves(row, col)
            self.update_board_ui()
            
        # If piece is selected and clicked on valid move
        elif self.selected_square and (row, col) in self.possible_moves:
            if self.try_make_move(self.selected_square, (row, col)):
                # Move succeeded
                self.selected_square = None
                self.possible_moves = []
                self.update_board_ui()
                
                if not self.game_over:  # Only start AI turn if game isn't over
                    # Lock board before AI's turn
                    self.lock_board()
                    # AI's turn
                    self.status_label.setText("AI thinking...")
                    self.status_label.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
                    self.ai_worker = AIWorkerThread(self.game_state.get_board_state())
                    self.ai_worker.move_ready.connect(self.handle_ai_move)
                    self.ai_worker.error.connect(self.handle_error)
                    self.ai_worker.start()
            else:
                # Move failed - error message already set by try_make_move
                self.update_board_ui()
            
        # Deselect current piece
        else:
            self.selected_square = None
            self.possible_moves = []
            self.update_board_ui()

    def try_make_move(self, from_pos, to_pos):
        """Attempt to make a move and return success status"""
        try:
            if self.game_over:
                return False  # Prevent moves after game is over
            
            # If this is a continuation of a multiple capture sequence,
            # only allow moves from the piece that just captured
            if self.selected_square and self.selected_square != from_pos:
                captures = self.game_state._get_capture_moves(self.selected_square[0], self.selected_square[1])
                if captures:
                    self.status_label.setText("Must complete capture sequence!")
                    self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
                    return False

            # Check if any captures are available
            captures = self.game_state._get_all_captures()
            if captures and from_pos not in captures:
                self.status_label.setText("Capture move available! Must take capture.")
                self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
                return False

            # If selected piece has captures, validate the move is a capture
            if captures:
                valid_moves = self.game_state.get_valid_moves(from_pos[0], from_pos[1])
                if to_pos not in valid_moves or abs(to_pos[0] - from_pos[0]) != 2:
                    self.status_label.setText("Must take capture move!")
                    self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
                    return False

            # Validate move
            if not self.game_state.is_valid_move(from_pos, to_pos):
                self.status_label.setText("Invalid move!")
                self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
                return False

            # Make the move
            moving_piece = self.game_state.board[from_pos[0]][from_pos[1]]
            self.game_state.board[from_pos[0]][from_pos[1]] = PieceType.EMPTY
    
            # Handle piece promotion for both colors
            if moving_piece == PieceType.RED and to_pos[0] == 0:
                moving_piece = PieceType.RED_KING
            elif moving_piece == PieceType.BLACK and to_pos[0] == 7:
                moving_piece = PieceType.BLACK_KING
    
            self.game_state.board[to_pos[0]][to_pos[1]] = moving_piece
    
            # Handle captures
            captured_piece = None
            if abs(to_pos[0] - from_pos[0]) == 2:
                captured_row = (from_pos[0] + to_pos[0]) // 2
                captured_col = (from_pos[1] + to_pos[1]) // 2
                captured_piece = self.game_state.board[captured_row][captured_col]
                self.game_state.board[captured_row][captured_col] = PieceType.EMPTY
        
                # After capture, check for additional captures
                additional_captures = self.game_state._get_capture_moves(to_pos[0], to_pos[1])
                if additional_captures:
                    # Don't switch turns and keep the piece selected for additional captures
                    self.selected_square = to_pos  # Update selected square to new position
                    self.possible_moves = additional_captures  # Update possible moves
                    self.status_label.setText("Additional capture required!")
                    self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
                    # Update move history but don't switch turns
                    self.game_state.move_history.append({
                        'from': from_pos,
                        'to': to_pos,
                        'piece': moving_piece,
                        'captured': captured_piece
                    })
                    return True
        
            # Switch turns only if no additional captures or after a non-capture move
            self.game_state.current_player = (PieceType.BLACK if self.game_state.current_player == PieceType.RED 
                                            else PieceType.RED)
        
            # Clear selected square and possible moves after turn is complete
            self.selected_square = None
            self.possible_moves = []
    
            # Update move history
            self.game_state.move_history.append({
                'from': from_pos,
                'to': to_pos,
                'piece': moving_piece,
                'captured': captured_piece
            })

            # Check for winner after move is complete
            if self.check_winner():
                return True  # Move was valid, but game is now over
            
            return True
        
        except Exception as e:
            print(f"Move error: {str(e)}")
            self.status_label.setText(f"Move error: {str(e)}")
            self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
            return False

    def handle_error(self, error_message):
        """Handle AI errors gracefully"""
        if not self.game_over:  # Only retry if game isn't over
            print(f"AI Error: {error_message}")
            self.status_label.setText("AI thinking... (retrying)")
            self.status_label.setStyleSheet("color: #1A1A1A; font-size: 16px; font-weight: bold;")
            # Keep board locked during retry
            self.lock_board()
            # Reset turn to AI and retry
            self.game_state.current_player = PieceType.BLACK
            self.ai_worker = AIWorkerThread(self.game_state.get_board_state())
            self.ai_worker.move_ready.connect(self.handle_ai_move)
            self.ai_worker.error.connect(self.handle_error)
            self.ai_worker.start()

    def handle_ai_move(self, move):
        """Handle AI move with proper error handling"""
        if self.game_over:
            return  # Prevent AI moves after game is over
            
        try:
            from_pos, to_pos = move
            if self.try_make_move(from_pos, to_pos):
                if not self.game_over:  # Only update status if game isn't over
                    self.status_label.setText("Your turn (Red)")
                    self.status_label.setStyleSheet("color: #FF4444; font-size: 16px; font-weight: bold;")
                    self.update_board_ui()
                    # Unlock board after successful AI move
                    self.unlock_board()
            else:
                self.retry_ai_move()
            
        except Exception as e:
            print(f"AI Move Error: {str(e)}")
            self.retry_ai_move()

    def retry_ai_move(self):
        """Helper method to retry AI move on failure"""
        if self.game_over:
            return
            
        print("AI move failed - retrying...")
        self.status_label.setText("AI thinking... (retrying)")
        self.status_label.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
    
        # Keep board locked during retry
        self.lock_board()
    
        # Clean up old worker and create new one
        if hasattr(self, 'ai_worker'):
            self.ai_worker.deleteLater()
        
        # Reset to AI's turn and start new worker
        self.game_state.current_player = PieceType.BLACK
        self.ai_worker = AIWorkerThread(self.game_state.get_board_state())
        self.ai_worker.move_ready.connect(self.handle_ai_move)
        self.ai_worker.error.connect(self.handle_error)
        self.ai_worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle("Fusion")
    
    # Create dark palette
    dark_palette = app.palette()
    dark_palette.setColor(QPalette.Window, QColor(26, 26, 26))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(dark_palette)
    
    window = CheckersGame()
    window.show()
    sys.exit(app.exec_())