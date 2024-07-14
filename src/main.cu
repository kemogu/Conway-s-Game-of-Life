#include "Game.cuh"

int main() {
    int width = 16;  // Örnek değerler
    int height = 16; // Örnek değerler
    Game game(width, height, 10.0f, 1.0f);
    game.start();
    return 0;
}
