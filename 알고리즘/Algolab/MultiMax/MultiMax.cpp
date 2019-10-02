#include <iostream>
#include <stdlib.h>
#include <ios>
using namespace std;

int compare(const void* value0, const void* value1);

int main(void) {
  cin.tie(NULL);
  ios::sync_with_stdio(false);

  int num_testcase = 0;
  cin >> num_testcase;

  for (int i=0;i<num_testcase;i++) {
    int length = 0;
    cin >> length;

    int numbers[length];
    for (int j=0;j<length;j++) {
      cin >> numbers[j];
    }

    qsort(numbers, length, sizeof(int), compare);

    int positive_index = -1;
    for (int j=0;j<length;j++) {
      if (numbers[j] > 0) {
        positive_index = j;
        break;
      }
    }

    int result = 0;
    // 양수가 없는 경우
    if (positive_index == -1) {
      result = numbers[0] * numbers[1];
    }
    // 양수가 하나인 경우
    else if (positive_index == length-1) {
      result = numbers[positive_index] * numbers[0] * numbers[1];
    }
    // 양수가 둘인 경우
    else if (positive_index == length-2) {
      int temp_a = numbers[positive_index] * numbers[positive_index+1];
      int temp_b = numbers[0] * numbers[1] * numbers[positive_index+1];
      result = temp_a > temp_b ? temp_a : temp_b;
    }
    // 양수가 3개 이상인 경우
    else {
      int temp_a = numbers[0] * numbers[1] * numbers[length-1];
      int temp_b = numbers[length-1] * numbers[length-2] * numbers[length-3];
      result = temp_a > temp_b ? temp_a : temp_b;
    }

    cout << result << endl;
  }
}

int compare(const void* value0, const void* value1) {
	return *(int*)value0 - *(int*)value1;
}
