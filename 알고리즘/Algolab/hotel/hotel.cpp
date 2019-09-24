#include <iostream>
#include <fstream>
using namespace std;

int main(void) {
    ifstream is;
    is.open("input.txt");
    if(!is) {
      cerr << "파일 오픈에 실패하였습니다" << endl;
    }

    int num_testcase = 0;
    is  >> num_testcase;

    int height = 0;
    int width = 0;
    int customer_num = 0;

    for (int i=0;i<num_testcase;i++) {
      is >> height;
      is >> width;
      is >> customer_num;

      int yy = customer_num % height == 0 ? height : customer_num % height;
      int xx = customer_num % height == 0 ? customer_num/height : (customer_num / height) + 1;

      if (xx < 10) {
        if (width)
        cout << yy << 0 << xx << endl;
      } else {
        cout << yy << xx << endl;
      }
    }

    is.close();
}
