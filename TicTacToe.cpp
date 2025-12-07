#include <iostream>
#include <cstdlib>   // rand, srand
#include <ctime>     // time
#include <vector>    // std::vector
#include <climits>   // INT_MIN, INT_MAX
#include <algorithm> // std::max, std::min
#include <string>
#include <limits>    // std::numeric_limits
#include <utility>   // std::pair, std::make_pair
#include <cmath>     // std::log, std::sqrt
#include <cctype>    // toupper

using namespace std;

// ===============================
// BASIC CONSTANTS AND GLOBAL DATA
// ===============================

const char PLAYER   = 'X';
const char COMPUTER = 'O';
const int  BOARD_SIZE = 3;

// The main tic-tac-toe board.
char board[BOARD_SIZE][BOARD_SIZE] = {
    { ' ', ' ', ' ' },
    { ' ', ' ', ' ' },
    { ' ', ' ', ' ' }
};

// To remember where each side moved.
vector<pair<int, int> > playerMoves;
vector<pair<int, int> > computerMoves;

// A move is just a row/column pair.
typedef pair<int, int> Move;

// Forward declarations
// --------------------

// Game / board helpers
void resetBoard();
void printBoard();
void printPastMoves(char mode);
int  countFreeSpaces(const char b[BOARD_SIZE][BOARD_SIZE]);
char checkWinner(const char b[BOARD_SIZE][BOARD_SIZE]);
void printWinnerMessage(char winner, char chosenMode);

// Input helpers
void handlePlayerMove(char playerChar, vector<pair<int,int> >& moveLog);
void playerMove();
void player2Move();

// Difficulty / AI selection
char selectComputerDifficulty();
Move getRandomComputerMove(const char currentBoard[BOARD_SIZE][BOARD_SIZE]);
int  getMctsIterationsForDifficulty(char aiChoice);

// Minimax
int  minimax(char currentBoard[BOARD_SIZE][BOARD_SIZE],
             int depth, bool isMaximizing,
             int alpha, int beta);
void minimaxMove();

// ===============================
// MCTS STRUCT AND FUNCTIONS
// ===============================

struct MCTSNode;
typedef MCTSNode* NodePtr;

// Node in the Monte Carlo Tree.
struct MCTSNode {
    char boardState[BOARD_SIZE][BOARD_SIZE]; // board at this node
    char playerToMove;                       // whose turn at this node
    Move lastMove;                           // move that led to this node

    int W;  // number of simulations that resulted in a COMPUTER win
    int N;  // number of times this node was visited

    NodePtr parent;
    vector<NodePtr> children;
    vector<Move> untriedMoves; // legal moves we haven’t expanded yet

    MCTSNode(const char b[BOARD_SIZE][BOARD_SIZE],
             NodePtr p,
             const Move& lm,
             char nextPlayer)
        : W(0),
          N(0),
          parent(p),
          lastMove(lm),
          playerToMove(nextPlayer)
    {
        // Copy board and gather all empty cells as untried moves.
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                boardState[i][j] = b[i][j];
                if (b[i][j] == ' ') {
                    untriedMoves.push_back(make_pair(i, j));
                }
            }
        }
    }

    ~MCTSNode() {
        // Recursively delete children to avoid memory leaks.
        for (size_t i = 0; i < children.size(); ++i) {
            delete children[i];
        }
    }
};

/**
 * Calculate UCT (Upper Confidence Bound for Trees) for a child node.
 * This balances:
 *   - exploitation: how good this node has been so far
 *   - exploration: how much we still need to try it
 */
double calculateUCT(NodePtr node, int parentVisits) {
    const double C = 1.414; // exploration constant ~ sqrt(2)
    if (node->N == 0) {
        // If node was never visited, treat it as extremely promising.
        return INT_MAX;
    }
    double winRate     = static_cast<double>(node->W) / node->N;
    double exploration = C * sqrt(log(static_cast<double>(parentVisits)) / node->N);
    return winRate + exploration;
}

/**
 * Among all children, pick the one with the highest UCT value.
 */
NodePtr selectBestChild(NodePtr node) {
    NodePtr bestChild = NULL;
    double bestUCT = -1.0;

    for (size_t i = 0; i < node->children.size(); ++i) {
        NodePtr child = node->children[i];
        double uct = calculateUCT(child, node->N);

        if (uct > bestUCT) {
            bestUCT = uct;
            bestChild = child;
        }
    }
    return bestChild;
}

/**
 * Expand by taking one untried move and creating a new child node.
 */
NodePtr expandNode(NodePtr node) {
    // Take a move from the end of the vector.
    Move move = node->untriedMoves.back();
    node->untriedMoves.pop_back();

    // Copy board.
    char newBoard[BOARD_SIZE][BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            newBoard[i][j] = node->boardState[i][j];
        }
    }

    // Apply the move for the current player.
    newBoard[move.first][move.second] = node->playerToMove;

    // Next player is the other side.
    char nextPlayer = (node->playerToMove == PLAYER ? COMPUTER : PLAYER);

    NodePtr child = new MCTSNode(newBoard, node, move, nextPlayer);
    node->children.push_back(child);
    return child;
}

/**
 * Play random moves until the game ends and return:
 *   10  -> COMPUTER wins
 *   -10 -> PLAYER wins
 *    0  -> draw
 */
int simulateRandomGame(const char startBoard[BOARD_SIZE][BOARD_SIZE], char playerToMove) {
    // Work on a temporary copy.
    char tempBoard[BOARD_SIZE][BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            tempBoard[i][j] = startBoard[i][j];
        }
    }

    char currentPlayer = playerToMove;

    while (true) {
        char winner = checkWinner(tempBoard);
        if (winner != ' ') {
            if (winner == COMPUTER) return 10;
            if (winner == PLAYER)   return -10;
            return 0; // 'D' draw
        }

        // Collect all empty spots.
        vector<Move> freeSpaces;
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (tempBoard[i][j] == ' ') {
                    freeSpaces.push_back(make_pair(i, j));
                }
            }
        }

        if (freeSpaces.empty()) {
            return 0; // No more moves -> draw.
        }

        // Pick a random empty cell.
        int randomIndex = rand() % freeSpaces.size();
        Move randomMove = freeSpaces[randomIndex];
        tempBoard[randomMove.first][randomMove.second] = currentPlayer;

        // Switch turn.
        currentPlayer = (currentPlayer == PLAYER ? COMPUTER : PLAYER);
    }
}

/**
 * Backpropagate the simulation result up the tree,
 * updating visit counts (N) and win counts (W) for COMPUTER.
 */
void backpropagate(NodePtr node, int result) {
    NodePtr current = node;

    while (current != NULL) {
        current->N++;

        // We only count COMPUTER wins as "wins".
        int scoreToAdd = 0;
        if (result == 10) {
            scoreToAdd = 1; // COMPUTER won this simulation.
        } else if (result == -10) {
            scoreToAdd = 0; // COMPUTER lost.
        }
        current->W += scoreToAdd;

        current = current->parent;
    }
}

/**
 * Run the full MCTS algorithm for a chosen number of iterations,
 * and return the move the computer will play.
 */
Move runMCTS(const char currentBoard[BOARD_SIZE][BOARD_SIZE], int iterations) {
    // Root node: it’s COMPUTER’s turn to move.
    NodePtr root = new MCTSNode(currentBoard, NULL, make_pair(-1, -1), COMPUTER);

    for (int i = 0; i < iterations; ++i) {
        NodePtr node = root;

        // ==== 1) SELECTION ====
        // Go down the tree while the node is fully expanded (no untried moves)
        // and the game is not over.
        while (node->untriedMoves.empty() &&
               checkWinner(node->boardState) == ' ' &&
               !node->children.empty()) {
            node = selectBestChild(node);
            if (node == NULL) {
                break;
            }
        }

        // ==== 2) EXPANSION ====
        if (!node->untriedMoves.empty() && checkWinner(node->boardState) == ' ') {
            node = expandNode(node);
        }

        // ==== 3) SIMULATION (ROLLOUT) ====
        char winner = checkWinner(node->boardState);
        int result = 0;

        if (winner == ' ') {
            // Game not finished, simulate to the end.
            result = simulateRandomGame(node->boardState, node->playerToMove);
        } else {
            // Terminal state at this node.
            if (winner == COMPUTER)      result = 10;
            else if (winner == PLAYER)   result = -10;
            else                         result = 0;
        }

        // ==== 4) BACKPROPAGATION ====
        backpropagate(node, result);
    }

    // After all simulations, pick the child with the most visits.
    NodePtr bestChild = NULL;
    int maxVisits = -1;

    for (size_t i = 0; i < root->children.size(); ++i) {
        NodePtr child = root->children[i];
        if (child->N > maxVisits) {
            maxVisits = child->N;
            bestChild = child;
        }
    }

    Move bestMove = make_pair(-1, -1);
    if (bestChild != NULL) {
        bestMove = bestChild->lastMove;
    }

    delete root; // this also deletes the entire tree.
    return bestMove;
}

/**
 * Wrapper for computer’s MCTS move.
 */
void mctsMove(int iterations) {
    cout << "Computer is thinking (MCTS with " << iterations
         << " simulations)..." << endl;

    Move bestMove = runMCTS(board, iterations);

    if (bestMove.first != -1) {
        board[bestMove.first][bestMove.second] = COMPUTER;
        computerMoves.push_back(bestMove);
    } else {
        cout << "Error: MCTS could not find a valid move." << endl;
    }
}

// ===============================
// MINIMAX IMPLEMENTATION (HARD)
// ===============================

/**
 * Minimax with alpha-beta pruning.
 * Returns:
 *   +10 if COMPUTER is winning
 *   -10 if PLAYER is winning
 *    0  for draw or equal outcome
 */
int minimax(char currentBoard[BOARD_SIZE][BOARD_SIZE],
            int depth,
            bool isMaximizing,
            int alpha,
            int beta)
{
    char winner = checkWinner(currentBoard);

    if (winner == COMPUTER) return 10;
    if (winner == PLAYER)   return -10;
    if (winner == 'D')      return 0;

    if (isMaximizing) {
        int bestScore = INT_MIN;

        // COMPUTER's turn: try all moves.
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (currentBoard[i][j] == ' ') {
                    currentBoard[i][j] = COMPUTER;
                    int score = minimax(currentBoard, depth + 1, false, alpha, beta);
                    currentBoard[i][j] = ' ';

                    bestScore = max(bestScore, score);
                    alpha = max(alpha, score);

                    if (beta <= alpha) {
                        // Cut off branch.
                        return bestScore;
                    }
                }
            }
        }
        return bestScore;
    } else {
        int bestScore = INT_MAX;

        // PLAYER's turn: try all moves.
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (currentBoard[i][j] == ' ') {
                    currentBoard[i][j] = PLAYER;
                    int score = minimax(currentBoard, depth + 1, true, alpha, beta);
                    currentBoard[i][j] = ' ';

                    bestScore = min(bestScore, score);
                    beta = min(beta, score);

                    if (beta <= alpha) {
                        // Cut off branch.
                        return bestScore;
                    }
                }
            }
        }
        return bestScore;
    }
}

/**
 * Use minimax to choose and play the best possible move for the computer.
 */
void minimaxMove() {
    cout << "Computer thinking..." << endl;

    int bestScore = INT_MIN;
    Move bestMove = make_pair(-1, -1);

    // Try all possible moves.
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board[i][j] == ' ') {
                board[i][j] = COMPUTER;
                int score = minimax(board, 0, false, INT_MIN, INT_MAX);
                board[i][j] = ' ';

                if (score > bestScore) {
                    bestScore = score;
                    bestMove = make_pair(i, j);
                }
            }
        }
    }

    if (bestMove.first != -1) {
        board[bestMove.first][bestMove.second] = COMPUTER;
        computerMoves.push_back(bestMove);
    } else {
        cout << "Error: could not find a valid move." << endl;
    }
}

// ===============================
// BASIC BOARD / GAME FUNCTIONS
// ===============================

void resetBoard() {
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            board[i][j] = ' ';
        }
    }
}

void printBoard() {
    cout << "\n    1   2   3\n";
    cout << "  +---+---+---+\n";
    for (int i = 0; i < BOARD_SIZE; ++i) {
        cout << i + 1 << " | ";
        for (int j = 0; j < BOARD_SIZE; ++j) {
            cout << board[i][j] << " | ";
        }
        cout << "\n";
        cout << "  +---+---+---+\n";
    }
}

/**
 * Print move history for both sides.
 */
void printPastMoves(char mode) {
    cout << "\nPast Moves:\n";
    if (mode == '1') {
        // Player vs Player
        cout << "Player 1 (X): ";
        for (size_t i = 0; i < playerMoves.size(); ++i) {
            cout << "(" << playerMoves[i].first + 1
                 << "," << playerMoves[i].second + 1 << ") ";
        }
        cout << "\nPlayer 2 (O): ";
        for (size_t i = 0; i < computerMoves.size(); ++i) {
            cout << "(" << computerMoves[i].first + 1
                 << "," << computerMoves[i].second + 1 << ") ";
        }
    } else {
        // Player vs Computer
        cout << "Player (X): ";
        for (size_t i = 0; i < playerMoves.size(); ++i) {
            cout << "(" << playerMoves[i].first + 1
                 << "," << playerMoves[i].second + 1 << ") ";
        }
        cout << "\nComputer (O): ";
        for (size_t i = 0; i < computerMoves.size(); ++i) {
            cout << "(" << computerMoves[i].first + 1
                 << "," << computerMoves[i].second + 1 << ") ";
        }
    }
    cout << "\n";
}

/**
 * Count empty spaces on a board.
 */
int countFreeSpaces(const char b[BOARD_SIZE][BOARD_SIZE]) {
    int freeSpaces = 0;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (b[i][j] == ' ') {
                freeSpaces++;
            }
        }
    }
    return freeSpaces;
}

/**
 * Return:
 *   'X' if player X wins
 *   'O' if player O wins
 *   'D' if draw
 *   ' ' if game not finished yet
 */
char checkWinner(const char b[BOARD_SIZE][BOARD_SIZE]) {
    // Check rows.
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (b[i][0] != ' ' &&
            b[i][0] == b[i][1] &&
            b[i][1] == b[i][2]) {
            return b[i][0];
        }
    }

    // Check columns.
    for (int j = 0; j < BOARD_SIZE; ++j) {
        if (b[0][j] != ' ' &&
            b[0][j] == b[1][j] &&
            b[1][j] == b[2][j]) {
            return b[0][j];
        }
    }

    // Diagonals.
    if (b[0][0] != ' ' &&
        b[0][0] == b[1][1] &&
        b[1][1] == b[2][2]) {
        return b[0][0];
    }
    if (b[0][2] != ' ' &&
        b[0][2] == b[1][1] &&
        b[1][1] == b[2][0]) {
        return b[0][2];
    }

    // Check for draw.
    if (countFreeSpaces(b) == 0) {
        return 'D';
    }

    // Game is still ongoing.
    return ' ';
}

// ===============================
// INPUT HANDLING
// ===============================

/**
 * Generic function to handle one player's move (human).
 * Asks for row/column until a valid empty tile is chosen.
 */
void handlePlayerMove(char playerChar, vector<pair<int,int> >& moveLog) {
    int row = 0;
    int column = 0;

    while (true) {
        cout << "Enter Row and Column (1-3 1-3): ";

        if (!(cin >> row)) {
            cout << "Invalid input type. Please enter numbers only.\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        if (!(cin >> column)) {
            cout << "Invalid input type. Please enter numbers only.\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        if (row < 1 || row > BOARD_SIZE ||
            column < 1 || column > BOARD_SIZE) {
            cout << "Invalid range. Please enter numbers between 1 and "
                 << BOARD_SIZE << ".\n";
            continue;
        }

        int r = row - 1;
        int c = column - 1;

        if (board[r][c] != ' ') {
            cout << "Tile (" << row << "," << column
                 << ") is already taken. Try again.\n";
        } else {
            board[r][c] = playerChar;
            moveLog.push_back(make_pair(r, c));
            break;
        }
    }
}

void playerMove() {
    handlePlayerMove(PLAYER, playerMoves);
}

void player2Move() {
    handlePlayerMove(COMPUTER, computerMoves);
}

// ===============================
// DIFFICULTY & RANDOM MOVE
// ===============================

/**
 * Ask player which difficulty to use for the computer.
 * E = random, M = MCTS, H = Minimax
 */
char selectComputerDifficulty() {
    char aiChoice;

    while (true) {
        cout << "\nSelect Computer Difficulty:\n";
        cout << "R. Regular  (Random moves)\n";
        cout << "H. Hard (MCTS simulations)\n";
        cout << "I. Impossible   (Minimax algorithm)\n";
        cout << "Enter your choice (R/H/I): ";

        cin >> aiChoice;

        if (cin.fail()) {
            cout << "Invalid input. Please enter R, H, or I.\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        aiChoice = toupper(aiChoice);
        if (aiChoice == 'R' || aiChoice == 'H' || aiChoice == 'I') {
            break;
        }

        cout << "Invalid choice. Please enter R, H, or I.\n";
    }
    return aiChoice;
}

/**
 * For Easy mode: pick a random empty cell.
 */
Move getRandomComputerMove(const char currentBoard[BOARD_SIZE][BOARD_SIZE]) {
    vector<Move> freeSpaces;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (currentBoard[i][j] == ' ') {
                freeSpaces.push_back(make_pair(i, j));
            }
        }
    }

    if (freeSpaces.empty()) {
        return make_pair(-1, -1);
    }

    int randomIndex = rand() % freeSpaces.size();
    return freeSpaces[randomIndex];
}

/**
 * Change these numbers to adjust how "smart" Medium (MCTS) feels.
 * More simulations = stronger but slower.
 */
int getMctsIterationsForDifficulty(char aiChoice) {
    if (aiChoice == 'H') {
        // Medium: reasonable default. 
        return 10000; // Can change number of simulatioons
    }
    return 0;
}

// ===============================
// WINNER MESSAGE
// ===============================

void printWinnerMessage(char winner, char chosenMode) {
    if (winner == PLAYER) {
        if (chosenMode == '1') {
            cout << "Player 1 (X) wins!\n";
        } else {
            cout << "Congratulations! You win!\n";
        }
    } else if (winner == COMPUTER) {
        if (chosenMode == '1') {
            cout << "Player 2 (O) wins!\n";
        } else {
            cout << "Computer wins! Better luck next time!\n";
        }
    } else if (winner == 'D') {
        cout << "          IT'S A TIE!\n"
                "       |\\_,,,---,,_\n"
                "ZZZzz /,`.-'`'    -.  ;-;;,_\n"
                "     |,4-  ) )-,_. ,\\ (  `'-'\n"
                "    '---''(_/--'  `-'\\_)\n";
    } else {
        cout << "Game ended unexpectedly.\n";
    }
}

// ===============================
// MAIN FUNCTION / GAME LOOP
// ===============================

int main() {
    srand(static_cast<unsigned int>(time(NULL)));

    cout << " _   _      _             _             \n"
            "| | (_)    | |           | |            \n"
            "| |_ _  ___| |_ __ _  ___| |_ ___   ___ \n"
            "| __| |/ __| __/ _` |/ __| __/ _ \\ / _ \\\n"
            "| |_| | (__| || (_| | (__| || (_) |  __/\n"
            " \\__|_|\\___|\\__\\__,_|\\___|\\__\\___/ \\___|\n"
            "\n"
            "\"============= Tic Tac Toe =============\"\n\n";

    while (true) {
        char chosenMode;
        char aiChoice = ' ';
        char playAgain;

        // ---- Select game mode ----
        while (true) {
            cout << "\nSelect game mode:\n";
            cout << "1. Player vs Player\n";
            cout << "2. Player vs Computer\n";
            cout << "3. Quit\n";
            cout << "Enter your choice: ";

            cin >> chosenMode;

            if (cin.fail() ||
                (chosenMode != '1' && chosenMode != '2' && chosenMode != '3')) {
                cout << "Invalid choice. Please enter 1, 2, or 3.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                continue;
            }
            break;
        }

        if (chosenMode == '3') {
            cout << "Exiting the game. Thanks for playing! :D\n";
            return 0;
        }

        if (chosenMode == '1') {
            cout << "Mode chosen: Player vs Player\n";
        } else {
            cout << "Mode chosen: Player vs Computer\n";
            aiChoice = selectComputerDifficulty();
            if (aiChoice == 'I')      cout << "AI chosen: Impossible (Minimax).\n";
            else if (aiChoice == 'H') cout << "AI chosen: Hard (MCTS).\n";
            else                      cout << "AI chosen: Regular (Random).\n";
        }

        // ---- Inner loop: keep playing games in this mode until user stops ----
        do {
            resetBoard();
            playerMoves.clear();
            computerMoves.clear();

            char winner = ' ';
            char currentPlayer = PLAYER; // X always starts.

            // ---- Single game loop ----
            while (winner == ' ' && countFreeSpaces(board) > 0) {
                cout << "\n";
                printBoard();
                printPastMoves(chosenMode);

                if (chosenMode == '1') {
                    // Player vs Player
                    cout << "Current Turn: Player "
                         << (currentPlayer == PLAYER ? "1 (X)" : "2 (O)") << "\n";

                    if (currentPlayer == PLAYER) {
                        playerMove();
                    } else {
                        player2Move();
                    }
                } else {
                    // Player vs Computer
                    cout << "Current Turn: "
                         << (currentPlayer == PLAYER ? "Player (X)" : "Computer (O)")
                         << "\n";

                    if (currentPlayer == PLAYER) {
                        playerMove();
                    } else {
                        // Computer's turn: use difficulty setting.
                        if (aiChoice == 'I') {
                            // Hard: Minimax
                            minimaxMove();
                        } else if (aiChoice == 'H') {
                            // Medium: MCTS
                            int sims = getMctsIterationsForDifficulty(aiChoice);
                            mctsMove(sims);
                        } else {
                            // Easy: random move
                            Move randomMove = getRandomComputerMove(board);
                            if (randomMove.first != -1) {
                                board[randomMove.first][randomMove.second] = COMPUTER;
                                computerMoves.push_back(randomMove);
                            } else {
                                cout << "No valid move for computer.\n";
                            }
                        }
                    }
                }

                // Check winner after the move.
                winner = checkWinner(board);

                // If still no winner, switch turns.
                if (winner == ' ') {
                    currentPlayer = (currentPlayer == PLAYER ? COMPUTER : PLAYER);
                }
            }

            // ---- End of single game ----
            cout << "\n===================================\n";
            cout << "GAME OVER!\n";
            printBoard();
            printPastMoves(chosenMode);
            printWinnerMessage(winner, chosenMode);
            cout << "===================================\n\n";

            cout << "Do you want to play again in the current mode? (Y/N): ";
            cin >> playAgain;
            playAgain = toupper(playAgain);

            if (cin.fail()) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                playAgain = 'N';
            }
        } while (playAgain == 'Y');
    }

    return 0;
}
