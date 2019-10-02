package trucks;

import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

public class main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int num_testcase = sc.nextInt();
        for (int i=0;i<num_testcase;i++) {
            int num_trucks = sc.nextInt();
            int bridge_length = sc.nextInt();
            int weight = sc.nextInt();
            int trucks[] = new int[num_trucks];
            
            for (int j=0;j<num_trucks;j++) {
                trucks[j] = sc.nextInt();    
            }
            
            System.out.println(solution(bridge_length, weight, trucks));
        }
        
        sc.close();
    }
    
    public static int solution(int bridge_length, int weight, int[] truck_weights) {
        Queue<Truck> waiting = new LinkedBlockingQueue<Truck>();
        for (Integer truck_weight : truck_weights) {
        	waiting.add(new Truck(truck_weight));
        }

        Queue<Truck> on_bridge = new LinkedBlockingQueue<Truck>();
        
        int time = 0;
        
        // 큐가 빌 때까지 반복
        while (!waiting.isEmpty() || !on_bridge.isEmpty()) {
            // 시간 다 지난 애는 빼냄
        	time++;
        	for (Truck truck : on_bridge) {
        		truck.increaseTime();
        	}
        	
        	if (!on_bridge.isEmpty() && on_bridge.peek().getTime() == bridge_length) {
        		on_bridge.poll();
        	}
        	
            // 새로 들어 올 애를 버틸 수 있는지 확인
        	if (!waiting.isEmpty() && canAfford(on_bridge, waiting.peek(), weight)) {
            	// 가능하면 넣고
        		on_bridge.add(waiting.poll());
        	}

        }
        
        return time;
    }
    
    private static boolean canAfford(Queue<Truck> queue, Truck truck, int weight_limit) {
    	int total_weight = 0;
    	for (Truck target : queue) {
    		total_weight += target.weight;
    	}
    	
    	return (total_weight + truck.getWeight()) <= weight_limit;
    }

    private static class Truck {
    	private int weight;
    	private int time;
    	
    	private Truck(int weight) {
    		this.weight = weight;
    		this.time = 0;
    	}
    	
		public int getWeight() {
			return weight;
		}
		public void setWeight(int weight) {
			this.weight = weight;
		}
		public int getTime() {
			return time;
		}
		public void setTime(int time) {
			this.time = time;
		}
		
		public void increaseTime() {
			time += 1;
		}
    }
}