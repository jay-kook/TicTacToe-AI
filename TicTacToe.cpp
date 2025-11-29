#include <iostream>
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()
#include <vector>    // For storing past moves and node children
#include <climits>   // For INT_MIN and INT_MAX
#include <algorithm> // For std::max, std::min, std::random_shuffle
#include <string>    // For string manipulation
#include <limits>    // For cin.ignore, numeric_limits
#include <utility>   // For std::pair, std::make_pair
#include <cmath>     // For std::log and std::sqrt (used in UCT)
#include <cctype>    // For toupper

using namespace std;

// Define constants for players
const char PLAYER = 'X';
const char COMPUTER = 'O'; // Computer is also 'O'
const int BOARD_SIZE = 3;

// Define the board (using a global array for simplicity)
char board[BOARD_SIZE][BOARD_SIZE] = {
    {' ', ' ', ' '},
    {' ', ' ', ' '},
    {' ', ' ', ' '}};

// Store past moves for both players
vector<pair<int, int> > playerMoves;
vector<pair<int, int> > computerMoves;

// Function Declarations (AI)
typedef pair<int, int> Move; // For MCTS
// Minimax functions
int minimax(char currentBoard[BOARD_SIZE][BOARD_SIZE], int depth, bool isMaximizing, int alpha, int beta);
void minimaxMove(); // Wrapper for Minimax AI
// MCTS functions (UPDATED)
void mctsMove(int iterations);
Move runMCTS(const char currentBoard[BOARD_SIZE][BOARD_SIZE], int iterations);
char selectComputerMode(); // Select AI difficulty

// map difficulty to simulations
int mctsIterationsForDifficulty(char aiChoice);

// Function Declarations (Game Logic)
void resetBoard();
void printBoard();
void printPastMoves(char mode);
int checkFreeSpaces();
char checkWinner(char b[BOARD_SIZE][BOARD_SIZE]);
void printWinner(char winner, char chosenMode);
void handlePlayerInput(char playerChar, vector<pair<int, int> >& movesLog);
void playerMove();
void player2Move();

// ******************************************************
// --- MCTS IMPLEMENTATION ---
// ******************************************************

struct MCTSNode;
typedef MCTSNode* NodePtr;

struct MCTSNode {
    char boardState[BOARD_SIZE][BOARD_SIZE];
    char playerToMove;
    Move lastMove;
    int W;
    int N;
    NodePtr parent;
    vector<NodePtr> children;
    vector<Move> untriedMoves;

    MCTSNode(const char b[BOARD_SIZE][BOARD_SIZE], NodePtr p, const Move& lm, char player)
        : W(0), N(0), parent(p), lastMove(lm), playerToMove(player)
    {
        // Copy the board state and find untried moves
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                boardState[i][j] = b[i][j];
                if (b[i][j] == ' ') {
                    untriedMoves.push_back(make_pair(i, j));
                }
            }
        }
        // Randomize the order of untried moves for exploration diversity
        random_shuffle(untriedMoves.begin(), untriedMoves.end());
    }
    ~MCTSNode() {
        for (vector<NodePtr>::iterator it = children.begin(); it != children.end(); ++it) {
            delete *it;
        }
    }
};

/**
 * @brief Calculates the Upper Confidence Bound 1 (UCB1) value for node selection.
 */
double calculateUCT(NodePtr node, int parentVisits) {
    const double C = 1.414; // Exploration parameter (sqrt(2))
    if (node->N == 0) return INT_MAX;
    double exploitation = (double)node->W / node->N;
    double exploration = C * sqrt(log((double)parentVisits) / node->N);
    return exploitation + exploration;
}

/**
 * @brief Selects the best child node using the UCB1 metric.
 */
NodePtr selectChild(NodePtr node) {
    NodePtr bestChild = NULL;
    double bestUCT = -1.0;
    for (vector<NodePtr>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
        NodePtr child = *it;
        double uctValue = calculateUCT(child, node->N);
        if (uctValue > bestUCT) {
            bestUCT = uctValue;
            bestChild = child;
        }
    }
    return bestChild;
}

/**
 * @brief Expands the tree by creating a new child node from an untried move.
 */
NodePtr expandNode(NodePtr node) {
    Move move = node->untriedMoves.back();
    node->untriedMoves.pop_back();

    char newBoard[BOARD_SIZE][BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; ++i)
        for (int j = 0; j < BOARD_SIZE; ++j)
            newBoard[i][j] = node->boardState[i][j];
    newBoard[move.first][move.second] = node->playerToMove;

    char nextPlayer = (node->playerToMove == PLAYER) ? COMPUTER : PLAYER;
    NodePtr newChild = new MCTSNode(newBoard, node, move, nextPlayer);
    node->children.push_back(newChild);
    return newChild;
}

/**
 * @brief Simulates a game from the current board state until a terminal state is reached.
 * Returns 10 (Computer Win), -10 (Player Win), or 0 (Draw).
 */
int simulateGame(const char b[BOARD_SIZE][BOARD_SIZE], char playerToMove) {
    char tempBoard[BOARD_SIZE][BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE; ++i)
        for (int j = 0; j < BOARD_SIZE; ++j)
            tempBoard[i][j] = b[i][j];

    char currentPlayer = playerToMove;

    while (true) {
        char winner = checkWinner(tempBoard);
        if (winner != ' ') {
            if (winner == COMPUTER) return 10;
            if (winner == PLAYER) return -10;
            return 0;
        }

        vector<Move> freeSpaces;
        for (int i = 0; i < BOARD_SIZE; ++i)
            for (int j = 0; j < BOARD_SIZE; ++j)
                if (tempBoard[i][j] == ' ')
                    freeSpaces.push_back(make_pair(i, j));

        if (freeSpaces.empty()) return 0;

        int random_index = rand() % freeSpaces.size();
        Move random_move = freeSpaces[random_index];
        tempBoard[random_move.first][random_move.second] = currentPlayer;
        currentPlayer = (currentPlayer == PLAYER) ? COMPUTER : PLAYER;
    }
}

/**
 * @brief Updates the win (W) and visit (N) counts for the current node and all its ancestors.
 */
void backpropagate(NodePtr node, int result) {
    NodePtr current = node;
    while (current != NULL) {
        current->N++;

        int score_to_backpropagate = 0;
        if (result == 10) score_to_backpropagate = 1;      // Computer win in simulation
        else if (result == -10) score_to_backpropagate = 0; // Computer loss in simulation

        current->W += score_to_backpropagate;
        current = current->parent;
    }
}

/**
 * @brief Main MCTS function. Runs a given number of iterations.
 * UPDATED: now takes `iterations` as a parameter.
 */
Move runMCTS(const char currentBoard[BOARD_SIZE][BOARD_SIZE], int iterations) {
    NodePtr root = new MCTSNode(currentBoard, NULL, make_pair(-1, -1), COMPUTER);

    for (int i = 0; i < iterations; ++i) {
        NodePtr node = root;

        // 1. Selection
        while (node->untriedMoves.empty() && checkWinner(node->boardState) == ' ') {
            NodePtr selected = selectChild(node);
            if (selected == NULL) break;
            node = selected;
        }

        // 2. Expansion
        if (!node->untriedMoves.empty() && checkWinner(node->boardState) == ' ') {
            node = expandNode(node);
        }

        char winner = checkWinner(node->boardState);
        int result = 0;

        if (winner == ' ') {
            // 3. Simulation (Rollout)
            result = simulateGame(node->boardState, node->playerToMove);
        } else {
            // Terminal state
            if (winner == COMPUTER) result = 10;
            else if (winner == PLAYER) result = -10;
            else result = 0;
        }

        // 4. Backpropagation
        backpropagate(node, result);
    }

    // Final Move Selection: Choose highest visit count
    NodePtr bestChild = NULL;
    int maxVisits = -1;

    for (vector<NodePtr>::iterator it = root->children.begin(); it != root->children.end(); ++it) {
        NodePtr child = *it;
        if (child->N > maxVisits) {
            maxVisits = child->N;
            bestChild = child;
        }
    }

    Move bestMove = bestChild ? bestChild->lastMove : make_pair(-1, -1);
    delete root;

    return bestMove;
}

/**
 * @brief Wrapper for MCTS algorithm.
 * UPDATED: now takes `iterations`.
 */
void mctsMove(int iterations) {
    cout << "Computer is thinking (MCTS search, " << iterations << " sims)..." << endl;
    Move bestMove = runMCTS(board, iterations);

    if (bestMove.first != -1) {
        board[bestMove.first][bestMove.second] = COMPUTER;
        computerMoves.push_back(bestMove);
    } else {
        cout << "Error: No valid move found for the computer." << endl;
    }
}

/**
 * @brief Minimax algorithm to determine the best move for the computer.
 */
int minimax(char currentBoard[BOARD_SIZE][BOARD_SIZE], int depth, bool isMaximizing, int alpha, int beta)
{
    char winner = checkWinner(currentBoard);

    if (winner == COMPUTER) {
        return 10;
    } else if (winner == PLAYER) {
        return -10;
    } else if (winner == 'D') {
        return 0;
    }

    if (isMaximizing) {
        int bestScore = INT_MIN;
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (currentBoard[i][j] == ' ') {
                    currentBoard[i][j] = COMPUTER;
                    int score = minimax(currentBoard, depth + 1, false, alpha, beta);
                    currentBoard[i][j] = ' ';
                    bestScore = max(bestScore, score);
                    alpha = max(alpha, score);
                    if (beta <= alpha) {
                        return bestScore;
                    }
                }
            }
        }
        return bestScore;
    } else {
        int bestScore = INT_MAX;
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (currentBoard[i][j] == ' ') {
                    currentBoard[i][j] = PLAYER;
                    int score = minimax(currentBoard, depth + 1, true, alpha, beta);
                    currentBoard[i][j] = ' ';
                    bestScore = min(bestScore, score);
                    beta = min(beta, score);
                    if (beta <= alpha) {
                        return bestScore;
                    }
                }
            }
        }
        return bestScore;
    }
}

/**
 * @brief Calculates and executes the computer's move using the Minimax algorithm.
 */
void minimaxMove()
{
    cout << "Computer is thinking..." << endl;
    int bestScore = INT_MIN;
    pair<int, int> bestMove = make_pair(-1, -1);

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
        cout << "Error: No valid move found for the computer." << endl;
    }
}

void resetBoard()
{
    for (int i = 0; i < BOARD_SIZE; ++i)
        for (int j = 0; j < BOARD_SIZE; ++j)
            board[i][j] = ' ';
}

void printBoard()
{
    cout << "\n    1   2   3" << endl;
    cout << "  +---+---+---+" << endl;
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        cout << i + 1 << " | ";
        for (int j = 0; j < BOARD_SIZE; ++j)
        {
            cout << board[i][j] << " | ";
        }
        cout << endl;
        cout << "  +---+---+---+" << endl;
    }
}

void printPastMoves(char mode)
{
    cout << "\nPast Moves:" << endl;
    if (mode == '1')
    {
        cout << "Player 1 (X): ";
        for (vector<pair<int, int> >::iterator it = playerMoves.begin(); it != playerMoves.end(); ++it)
            cout << "(" << it->first + 1 << "," << it->second + 1 << ") ";
        cout << endl;

        cout << "Player 2 (O): ";
        for (vector<pair<int, int> >::iterator it = computerMoves.begin(); it != computerMoves.end(); ++it)
            cout << "(" << it->first + 1 << "," << it->second + 1 << ") ";
    }
    else
    {
        cout << "Player (X): ";
        for (vector<pair<int, int> >::iterator it = playerMoves.begin(); it != playerMoves.end(); ++it)
            cout << "(" << it->first + 1 << "," << it->second + 1 << ") ";
        cout << endl;

        cout << "Computer (O): ";
        for (vector<pair<int, int> >::iterator it = computerMoves.begin(); it != computerMoves.end(); ++it)
            cout << "(" << it->first + 1 << "," << it->second + 1 << ") ";
    }
    cout << endl;
}

int checkFreeSpaces()
{
    int freeSpaces = 0;
    for (int i = 0; i < BOARD_SIZE; ++i)
        for (int j = 0; j < BOARD_SIZE; ++j)
            if (board[i][j] == ' ')
                freeSpaces++;
    return freeSpaces;
}

char checkWinner(char b[BOARD_SIZE][BOARD_SIZE])
{
    for (int i = 0; i < BOARD_SIZE; i++)
        if (b[i][0] == b[i][1] && b[i][0] == b[i][2] && b[i][0] != ' ')
            return b[i][0];

    for (int i = 0; i < BOARD_SIZE; i++)
        if (b[0][i] == b[1][i] && b[0][i] == b[2][i] && b[0][i] != ' ')
            return b[0][i];

    if (b[0][0] == b[1][1] && b[0][0] == b[2][2] && b[0][0] != ' ')
        return b[0][0];
    if (b[0][2] == b[1][1] && b[0][2] == b[2][0] && b[0][2] != ' ')
        return b[0][2];

    int freeSpaces = 0;
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            if (b[i][j] == ' ')
                freeSpaces++;

    if (freeSpaces == 0)
        return 'D';

    return ' ';
}

void handlePlayerInput(char playerChar, vector<pair<int, int> >& movesLog)
{
    int row = 0, column = 0;

    while (true)
    {
        cout << "Enter Row and Column #(1-3): ";

        if (!(cin >> row)) {
            cout << "Invalid input type. Please enter numbers only." << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        if (!(cin >> column)) {
            cout << "Invalid input type. Please enter numbers only." << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        if (row < 1 || row > BOARD_SIZE || column < 1 || column > BOARD_SIZE)
        {
            cout << "Invalid range. Please enter numbers between 1 and " << BOARD_SIZE << "." << endl;
            continue;
        }

        int r = row - 1;
        int c = column - 1;

        if (board[r][c] != ' ')
        {
            cout << "Tile (" << row << "," << column << ") is already taken. Try again." << endl;
        }
        else
        {
            board[r][c] = playerChar;
            movesLog.push_back(make_pair(r, c));
            break;
        }
    }
}

void playerMove()
{
    handlePlayerInput(PLAYER, playerMoves);
}

void player2Move()
{
    handlePlayerInput(COMPUTER, computerMoves);
}

/**
 * difficulty menu: E / M / H
 */
char selectComputerMode() {
    char aiChoice;
    while (true) {
        cout << "\nSelect Computer Difficulty:" << endl;
        cout << "E. Easy   (MCTS - low simulations)" << endl;
        cout << "M. Medium (MCTS - more simulations)" << endl;
        cout << "H. Hard   (Minimax)" << endl;
        cout << "Enter your choice (E/M/H): ";

        cin >> aiChoice;

        if (cin.fail()) {
            cout << "Invalid input. Please enter E, M, or H." << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }

        aiChoice = toupper(aiChoice);

        if (aiChoice == 'E' || aiChoice == 'M' || aiChoice == 'H') {
            break;
        }

        cout << "Invalid choice. Please enter E, M, or H." << endl;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
    return aiChoice;
}

/**
 * Change these numbers anytime to rebalance difficulty.
 */
int mctsIterationsForDifficulty(char aiChoice) {
    if (aiChoice == 'E') return 200;   // Easy
    if (aiChoice == 'M') return 100000; // Medium
    return 0;                          // Hard uses minimax
}

void printWinner(char winner, char chosenMode)
{
    if (winner == PLAYER)
    {
        cout << (chosenMode == '1' ? "Player 1 (X) wins!" : "Congratulations! You win!") << endl;
    }
    else if (winner == COMPUTER)
    {
        cout << (chosenMode == '1' ? "Player 2 (O) wins!" : "Computer wins! Better luck next time!") << endl;
    }
    else if (winner == 'D')
    {
        cout << "          IT'S A TIE!\n"
                "       |\\_,,,---,,_\n"
                "ZZZzz /,`.-'`'    -.  ;-;;,_\n"
                "     |,4-  ) )-,_. ,\\ (  `'-'\n"
                "    '---''(_/--'  `-'\\_)" << endl;
    }
    else
    {
        cout << "Game ended unexpectedly without a clear result." << endl;
    }
}

int main()
{
    char playAgain;
    srand(time(nullptr));

    cout << " _   _      _             _             \n"
            "| | (_)    | |           | |            \n"
            "| |_ _  ___| |_ __ _  ___| |_ ___   ___ \n"
            "| __| |/ __| __/ _` |/ __| __/ _ \\ / _ \\\n"
            "| |_| | (__| || (_| | (__| || (_) |  __/\n"
            " \\__|_|\\___|\\__\\__,_|\\___|\\__\\___/ \\___|\n"
            "\n"
            "\"============= Tic Tac Toe =============\"\n"
         << endl;

    do
    {
        char chosenMode;
        char aiChoice = ' ';

        while (true)
        {
            cout << "\nSelect game mode:" << endl;
            cout << "1. Player vs Player" << endl;
            cout << "2. Player vs Computer" << endl;
            cout << "3. Quit" << endl;
            cout << "Enter your choice: ";

            cin >> chosenMode;

            if (cin.fail() || (chosenMode != '1' && chosenMode != '2' && chosenMode != '3')) {
                cout << "Invalid choice. Please enter 1, 2, or 3." << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                continue;
            }
            break;
        }

        if (chosenMode == '3')
        {
            cout << "Exiting the game. Thanks for playing! :D" << endl;
            return 0;
        }

        cout << "Mode chosen: " << (chosenMode == '1' ? "Player vs Player" : "Player vs Computer (AI)") << endl;

        if (chosenMode == '2') {
            aiChoice = selectComputerMode();
            if (aiChoice == 'H') cout << "AI chosen: Hard (Minimax).\n";
            else if (aiChoice == 'M') cout << "AI chosen: Medium (MCTS - more sims).\n";
            else cout << "AI chosen: Easy (MCTS - fewer sims).\n";
        }

        do
        {
            resetBoard();
            playerMoves.clear();
            computerMoves.clear();

            char winner = ' ';
            char currentPlayer = PLAYER;

            do
            {
                cout << endl;
                cout << "\n" << endl;
                printBoard();
                printPastMoves(chosenMode);

                if (chosenMode == '1') // PvP
                {
                    cout << "Current Turn: Player " << (currentPlayer == PLAYER ? "1 (X)" : "2 (O)") << endl;
                    if (currentPlayer == PLAYER) playerMove();
                    else player2Move();
                }
                else // PvC
                {
                    cout << "Current Turn: " << (currentPlayer == PLAYER ? "Player (X)" : "Computer (O)") << endl;
                    if (currentPlayer == PLAYER)
                    {
                        playerMove();
                    }
                    else
                    {
                        // *** UPDATED AI Selection Logic ***
                        if (aiChoice == 'H') {
                            minimaxMove();
                        } else {
                            int sims = mctsIterationsForDifficulty(aiChoice);
                            mctsMove(sims);
                        }
                    }
                }

                winner = checkWinner(board);

                if (winner == ' ') {
                   currentPlayer = (currentPlayer == PLAYER) ? COMPUTER : PLAYER;
                }

            } while (winner == ' ' && checkFreeSpaces() > 0);

            cout << "\n===================================" << endl;
            cout << "GAME OVER!" << endl;
            printBoard();
            printPastMoves(chosenMode);
            printWinner(winner, chosenMode);
            cout << "===================================\n" << endl;

            cout << "Do you want to play again in the current mode? (Y/N): ";
            cin >> playAgain;

            if (cin.fail()) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                playAgain = 'n';
            }

        } while (playAgain == 'Y' || playAgain == 'y');

    } while (true);

    return 0;
}
