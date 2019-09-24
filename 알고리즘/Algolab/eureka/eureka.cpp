#include <iostream>
#include <fstream>
using namespace std;

int main(void) {
    ifstream is;
    is.open("input.txt");
    if(!is) {
      cerr << "파일 오픈에 실패하였습니다" << endl;
    }

    int tnum[45];
    for (int i=1; i<45; i++) {
      tnum[i] = i*(i+1) / 2;
    }

    int num_testcase = 0;
    is  >> num_testcase;

    int num_test = 0;

    for (int n=0;n<num_testcase;n++) {
      is >> num_test;
      bool flag = false;

      for (int i=1; i<45; i++) {
        for (int j=1; j<45; j++) {
          for (int k=1; k<45; k++) {
            if (tnum[i] + tnum[j] + tnum[k] == num_test) {
              flag = true;
              break;
            }
          }
          if (flag) break;
        }
        if (flag) break;
      }

      if (flag) {
        cout << 1 << endl;
      } else {
        cout << 0 << endl;
      }
    }

    is.close();
}
