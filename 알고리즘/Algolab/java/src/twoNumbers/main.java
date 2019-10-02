package twoNumbers;

import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int num_testcase = sc.nextInt();
		
		for (int i=0; i<num_testcase;i++) {

			
			int len = sc.nextInt();
			int target = sc.nextInt();
			ArrayList<Integer> numbers = new ArrayList<Integer>();
			
			for  (int j=0;j<len;j++) {
				int num = sc.nextInt();
				numbers.add(num);
			}
			
			
			int min = target+1;
			int count = 1;
			
			for (int j=0;j<numbers.size()-1;j++) {
				int key = target - numbers.get(j);
				
				int temp = Math.abs(binarySearch(key, numbers, j) - key);
				
				if (min < temp) {
					continue;
				} else if (min == temp) {
					count += 1;
				} else {
					min = temp;
					count = 1;
				}
			} 
			
			System.out.println(count);
		}
		
		sc.close();
	}
	
	public static int binarySearch(int key, ArrayList<Integer> nums, int exclude) {
		ArrayList<Integer> tempNum = new ArrayList<Integer>();
		for (Integer num : nums) {
			tempNum.add(num);
		}
		
		Collections.sort(tempNum);
		for (int i=0;i<=exclude;i++) {
			tempNum.remove(0);
		}
		
		int mid = 0;
		int left = 0;
		int right = tempNum.size() -1;
		boolean success = false;
		
		while (right >= left) {
			mid = (right + left) / 2;
			
			if (key == tempNum.get(mid)) {
				success = true;
				break;
			}
			
			if (key < tempNum.get(mid)) {
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		
		int result = tempNum.get(mid);
		
		int[] cands = new int[3];
		cands[0] = (mid == 0) ? -1 : tempNum.get(mid-1);
		cands[1] = tempNum.get(mid);
		cands[2] = (mid == tempNum.size()-1) ? -1 : tempNum.get(mid+1);
		
		if (success) {
			return result;
		} else {
			int min = 1;
			for (int i=0;i<cands.length;i++) {
				if (cands[i] == -1) {
					continue;
				}
				if (Math.abs(cands[i] - key) < Math.abs(cands[min] - key)) {
					min = i;
				}
			}
			
			return cands[min];			
		}
	}
}
