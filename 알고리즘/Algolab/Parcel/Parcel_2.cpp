#include <iostream>
#include <ios>
#include <stdlib.h>
using namespace std;

int compare(const void* value0, const void* value1);

int main(void) {
  cin.tie(NULL);
  ios::sync_with_stdio(false);
  int num_testcase = 0;
  cin >> num_testcase;

  for (int i=0;i<num_testcase;i++) {
    int weight, num_items;
    cin >> weight >> num_items;

    int items[num_items];
    for (int j=0;j<num_items;j++) {
      cin >> items[j];
    }

    qsort(items, num_items, sizeof(int), compare);

    bool checked[weight+1];
    bool end = false;

    for (int j=0;j<weight+1;j++) {
      checked[j] = false;
    }

    for (int j=0;j<num_items;j++) {
      for (int k=j+1;k<num_items;k++) {
        if (items[j] + items[k] >= weight) {
          break;
        }
        // cout << weight << "] j : " << j << ", k : " << k <<
            // " = " << items[j] + items[k] << " pass test" << endl;
        if (checked[weight - items[j] - items[k]]) {
          // cout << "------------ check : " << items[j]+items[k] << endl;
          end = true;
          break;
        }
      }
      if (end) {
        break;
      }
      for (int l=0;l<j;l++) {
        int sum = items[l] + items[j];
        if (sum <= weight) {
          checked[items[l] + items[j]] = true;
        }
      }
    }
    if (end) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
}

int compare(const void* value0, const void* value1) {
	return *(int*)value0 - *(int*)value1;
}
