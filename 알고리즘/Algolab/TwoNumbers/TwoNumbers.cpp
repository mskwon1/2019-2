#include <iostream>
#include <stdlib.h>
#include <ios>
using namespace std;

int binarySearch(const int*, int, int, int, int&);
int compare(const void*, const void*);

int main(void) {
  cin.tie(NULL);
  ios::sync_with_stdio(false);

  int num_testcase = 0;
  cin >> num_testcase;

  for (int i=0;i<num_testcase;i++) {
    int length, number;
    cin >> length >> number;

    int numbers[length];
    for (int j=0;j<length;j++) {
      cin >> numbers[j];
    }

    // quick sort numbers array
    qsort(numbers, length, sizeof(int), compare);

    // minimum diff with target number
    int min_diff = 2147483647;
    // number of min_diffs
    int count = 0;

    // cout << "IND \t LEN \t TAR \t DIFF" << endl;
    for (int j=0;j<length-1;j++) {
      // target number to find through binary search
      int target = number - numbers[j];
      int temp_count = 0;
      // cout << "NUM : " << number << "j : " << numbers[j] << endl;
      int temp_diff = binarySearch(numbers, target, length, j, temp_count);

      if (temp_diff < min_diff) {
        min_diff = temp_diff;
        count = temp_count;
      } else if (temp_diff == min_diff) {
        count += temp_count;
      } else {
        continue;
      }
    }

    cout << count << endl;
  }
}

// returns least diff with target number in array
int binarySearch(const int* array, int target, int length, int index, int& temp_count) {
  int left = index + 1;
  int right = length - 1;
  int mid = (left + right) / 2;

  while (left <= right) {
    mid = (left + right) / 2;

    if (array[mid] > target) {
      right = mid - 1;
      continue;
    } else if (array[mid] < target) {
      left = mid + 1;
      continue;
    } else {
      break;
    }
  }

  int cands[3];
  cands[0] = ((mid-1) > index) ? array[mid-1] : -1;
  cands[1] = array[mid];
  cands[2] = ((mid+1) <= length-1) ? array[mid+1] : -1;

  int min = 2147483647;

  for (int i=0;i<3;i++) {
    if (cands[i] == -1) {
      continue;
    }

    int temp = abs(target - cands[i]);
    if (min > temp) {
      min = temp;
      temp_count = 1;
    } else if (min == temp) {
      temp_count += 1;
    }
  }

  // cout << index << " \t " << length << " \t " << target << " \t " << min << endl;

  return min;
}

int compare(const void* value1, const void* value2) {
	return *(int*)value1 - *(int*)value2;
}
