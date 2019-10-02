#include <iostream>
using namespace std;

void findGoldbachNumbers(const int*, const int, const int, int&, int&);

int main(void) {
  // prime_numbers에 1000까지의 소수 저장
  int prime_numbers[1000];
  int end_index = 0;

  for (int i=2;i<1000;i++) {
    bool prime = true;

    for (int j=2;j<=(i/2);j++) {
      if (i % j == 0) {
        prime = false;
        break;
      }
    }

    if (prime) {
      prime_numbers[end_index++] = i;
    }
  }

  int num_testcase = 0;
  cin >> num_testcase;

  for (int i=0;i<num_testcase;i++) {
    int number = 0;
    cin >> number;

    int num1 = 0;
    int num2 = 0;
    findGoldbachNumbers(prime_numbers, end_index, number, num1, num2);

    cout << num1 << " " << num2 << endl;
  }
}

// 소수 배열과 끝 인덱스를 받아 해당 소수들의 합이 number가 되는 num1, num2를 SET 해주는함수
void findGoldbachNumbers(const int* prime_numbers, const int end_index, const int number, int& num1, int& num2) {
  int tempNum1 = 0;
  int tempNum2 = 0;
  int diff = 1000;
  for (int i=0;i<end_index;i++) {
    if (prime_numbers[i] > number/2) break;

    for (int j=0;j<end_index;j++) {
      if (prime_numbers[j] > number) break;

      if (prime_numbers[i] + prime_numbers[j] == number) {
        tempNum1 = prime_numbers[i];
        tempNum2 = prime_numbers[j];

        if (tempNum2 - tempNum1 < diff) {
          diff = tempNum2 - tempNum1;
          num1 = tempNum1;
          num2 = tempNum2;
        }
      }
    }
  }
}
