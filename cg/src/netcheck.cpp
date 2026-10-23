// Prints the network outputs for feature rows on stdin: netcheck WEIGHTS.txt < rows (44 floats per line)
#define main bot_main
#include BOT_SOURCE
#undef main
#include <fstream>
using namespace mpr;
int main(int argc, char** argv) {
  std::ifstream f(argv[1]);
  std::string blob((std::istreambuf_iterator<char>(f)), {});
  MLP m;
  if (!dequantize(z85Decode(blob.c_str()), m)) return 1;
  float x[N_FULL], y[BC_OUT];
  while (true) {
    for (int i = 0; i < N_FULL; ++i)
      if (!(std::cin >> x[i])) return 0;
    m.forward(x, y);
    for (int i = 0; i < BC_OUT; ++i) std::printf("%.4f ", y[i]);
    std::printf("\n");
  }
}
