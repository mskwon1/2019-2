#include <iostream>
using namespace std;

int happy(int);

int main(void) {
  int num_testcase = 0;
  cin >> num_testcase;
  for (int i=0;i<num_testcase;i++) {
    int n = 0;
    cin >> n;
    if (happy(n)) {
      cout << "HAPPY" << endl;
    } else {
      cout << "UNHAPPY" << endl;
    }
  }
}

int happy(int n) {
  if (n == 1) {
    return 1;
  } else if (n == 4) {
    return 0;
  } else {
    int sum = 0;

    while (n >= 10) {
        int digit = n % 10;
        sum += digit*digit;
        n /= 10;
    }

    sum += n*n;

    return happy(sum);
  }
}
