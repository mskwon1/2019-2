#include <iostream>
using namespace std;

long long getExpectation(long long, long long);

int main(void) {
    int num_testcase = 0;
    cin  >> num_testcase;

    for (int i=0;i<num_testcase;i++) {
      int num_teams = 0;
      int num_matches = 0;

      cin >> num_teams;
      long long plus[num_teams];
      long long minus[num_teams];
      long long expectation[num_teams];

      for (int j=0;j<num_teams;j++) {
        plus[j] = 0;
        minus[j] = 0;
        expectation[j] = 0;
      }

      cin >> num_matches;
      for (int j=0;j<num_matches;j++) {
        int home = 0;
        int away = 0;
        int home_score = 0;
        int away_score = 0;

        cin >> home;
        cin >> away;
        cin >> home_score;
        cin >> away_score;

        plus[home-1] += home_score;
        minus[home-1] += away_score;

        plus[away-1] += away_score;
        minus[away-1] += home_score;
      }

      long long max = 0;
      long long min = 20000;
      for (int j=0;j<num_teams;j++) {
        expectation[j] = getExpectation(plus[j], minus[j]);
        // cout << plus[j] << " : " << minus[j] << " : " << expectation[j] << endl;

        if (expectation[j] > max) {
          max = expectation[j];
        }

        if (expectation[j] < min) {
          min = expectation[j];
        }
      }

      cout << max << endl;
      cout << min << endl;
    }
}

long long getExpectation(long long plus, long long minus) {
  if (plus == 0 && minus == 0) {
    return 0;
  } else {
    long long result = (plus*plus*1000) / (plus*plus + minus*minus);
    return result;
  }
}
