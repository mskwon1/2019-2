package henry;

import java.util.Scanner;

public class Henry {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int num_testcase = sc.nextInt();
		
		for (int i=0;i<num_testcase;i++) {
			int a = sc.nextInt();
			int b = sc.nextInt();
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
			
			System.out.println(b);
		
		}
	}
	
	public static int gcd(int a, int b) {
		if (b == 0) {
			return a;
		} else {
			return gcd(b, a % b);
		}
	}
}
