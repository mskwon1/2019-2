#include <iostream>
using namespace std;

int changeNum(int, int*, int);
bool testPalindrome(const int*, int);

int main(void) {
  int num_testcase = 0;
  cin >> num_testcase;

  for (int i=0;i<num_testcase;i++) {
    int number = 0;
    cin >> number;
    bool success = false;

    for (int j=2;j<=64;j++) {
      int numArray[20];
      int end_index = 0;

      end_index = changeNum(number, numArray, j);

      if (testPalindrome(numArray, end_index)) {
        success = true;
        break;
      }
    }

    cout << success << endl;
  }
}

// number를 base진법으로 바꿔 그 결과를 한자리씩 numArray에 저장, 값이 들어간 index+1를 리턴
int changeNum(int number, int* numArray, int base) {
  int index = 0;
  while (number >= base) {
    numArray[index++] = number % base;
    number /= base;
  }
  numArray[index++] = number;

  return index;
}

// 해당 배열의 end_index-1번째까지 Palindrome인지 확인 후 bool값 리턴
bool testPalindrome(const int* numArray, int end_index) {
  for (int i=0;i<(end_index/2);i++) {
    if (numArray[i] != numArray[end_index-i-1]) {
      return false;
    }
  }

  return true;
}
