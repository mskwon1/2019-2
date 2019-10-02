#include <iostream>
#include <stdlib.h>
#include <ios>
using namespace std;

void find_nearest_point(const int* set, const int key, const int start, const int end,
	int* local_minimum_difference, int* local_minimum_count);
int compare(const void* value0, const void* value1);

int main(int argc, char* argv[]) {
  cin.tie(NULL);
  ios::sync_with_stdio(false);

	int* set = NULL;

	int minimum_difference = 0;
	int minimum_count = 0;

	int local_minimum_difference = 0;
	int local_minimum_count = 0;

	int n = 0, z = 0;
	int q = 0, c = 0;

	int key = 0, start = 0, end = 0;

	int i, j, tmp, flag_update;

	cin >> n;
	for (z=0; z<n; z++) {
		cin >> q >> c;

		set = (int*)malloc(sizeof(int)*q);

		for (i=0; i<q; i++) {
			cin >> set[i];
		}

		qsort(set, q, sizeof(int), compare);

		for (i=0; i<q-1; i++) {
			{
				key = c - set[i];
				start = i+1;
				end = q-1;

				local_minimum_difference = -1;
				local_minimum_count = 0;

				flag_update = 0;
			}

			find_nearest_point(set, key, start, end, &local_minimum_difference, &local_minimum_count);

			if (i == 0) {
				minimum_difference = local_minimum_difference;
				minimum_count = local_minimum_count;
				flag_update = 1;
			} else {
				if (local_minimum_difference == minimum_difference) {
					minimum_count += local_minimum_count;
				}
				if (local_minimum_difference > minimum_difference) {

				}
				if (local_minimum_difference < minimum_difference) {
					minimum_difference = local_minimum_difference;
					minimum_count = local_minimum_count;
					flag_update = 1;
				}
			}
		}
		cout << minimum_count << endl;

		if (set != NULL) {
			free(set);
		}
	}

	return 0;
}

void find_nearest_point(const int* set, const int key, const int start, const int end,
						int* local_minimum_difference, int* local_minimum_count) {

	int left = start;
	int right = end;
	int mid = 0;

	int tmp[3] = {0, };

	int i;

	if (left > right) {
		return;
	}

	while (left <= right) {
		mid = (left + right) / 2;
		if (set[mid] == key) {
			break;
		}
		if (key < set[mid]) {
			right = mid - 1;
			continue;
		}
		if (key > set[mid]) {
			left = mid + 1;
			continue;
		}
	}

	tmp[0] = (mid - 1 >= start) ? abs(key - set[mid - 1] ) : -1;
	tmp[1] = abs(key - set[mid]);
	tmp[2] = (mid + 1 <= end) ? abs(key - set[mid + 1] ) : -1;

	if (set[mid] == key) {
		*local_minimum_difference = tmp[1];
		*local_minimum_count = 1;
		return;
	}

	*local_minimum_difference = tmp[1];
	*local_minimum_count = 0;
	for (i=0; i<3; i++) {
		if (tmp[i] != -1 && tmp[i] < *local_minimum_difference) {
			*local_minimum_difference = tmp[i];
		}
	}
	for (i=0; i<3; i++) {
		if (tmp[i] == *local_minimum_difference) {
			(*local_minimum_count)++;
		}
	}
}

int compare(const void* value0, const void* value1) {
	return *(int*)value0 - *(int*)value1;
}
