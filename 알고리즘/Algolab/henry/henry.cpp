#include <iostream>
using namespace std;

int gcd(int, int);

int main(void) {

		int num_testcase = 0;
		cin >> num_testcase;

		for (int i=0;i<num_testcase;i++) {
			int a = 0;
			int b = 0;
			cin >> a;
			cin >> b;

			int x = 0;

			while (a != 1) {
				if (b % a == 0) {
					x = b / a;
				} else {
					x = b / a + 1;
				}

				a = a * x - b;
				b *= x;

				int g = gcd(a, b);
				a /= g;
				b /= g;
			}

			cout << b << endl;

		}
}

int gcd(int a, int b) {
	if (b == 0) {
		return a;
	} else {
		return gcd(b, a % b);
	}
}
