#include <iostream>
#include <ios>
#include <set>
using namespace std;

int main(void) {
    cin.tie(NULL);
    ios::sync_with_stdio(false);
    int num_testcase = 0;
    cin  >> num_testcase;

    for (int i=0;i<num_testcase;i++) {
      // 값 입력 받기
      int horz = 0;
      int vert = 0;

      cin >> vert;
      cin >> horz;

      long long result = 0;
      long long boxes[vert][horz];
      set<long long> max_values;

      for (int j=0;j<vert;j++) {
        for (int k=0;k<horz;k++) {
          cin >> boxes[j][k];
          result += boxes[j][k];
        }
      }

      // 최대값을 넣을 SET
      set<long long>::iterator iter;

      // 최대값 추출
      for (int j=0;j<horz;j++) {
        int max = 0;
        for (int k=0;k<vert;k++) {
          if (boxes[k][j] > max) {
            max = boxes[k][j];
          }
        }
        max_values.insert(max);
      }
      for (int j=0;j<vert;j++) {
        int max = 0;
        for (int k=0;k<horz;k++) {
          if (boxes[j][k] > max) {
            max = boxes[j][k];
          }
        }
        max_values.insert(max);
      }

      // 결과 계산
      for(iter=max_values.begin(); iter!=max_values.end(); ++iter) {
        result -= *iter;
      }

      cout << result << endl;
    }
}
