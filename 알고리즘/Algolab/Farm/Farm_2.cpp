#include <iostream>
using namespace std;

int main(void) {
  int a, b, n, w;
  cin >> a >> b >> n >> w;

  int sheep = 0;
  int wolf = 0;
  int count = 0;

  // 루프 돌면서 해 찾기
  for (int temp_sheep=1, temp_wolf=n-1; temp_sheep<n; temp_sheep++, temp_wolf--) {
    if (temp_sheep*a + b*temp_wolf == w) {
      sheep = temp_sheep;
      wolf = temp_wolf;
      count++;
    }

    // 해 2개 이상이면 break
    if (count > 1) {
      break;
    }
  }

  // 해가 하나면 양, 늑대 값 출력, 아니면 -1 출력
  if (count == 1) {
    cout << sheep << " " << wolf << endl;
  } else {
    cout << -1 << endl;
  }
}
