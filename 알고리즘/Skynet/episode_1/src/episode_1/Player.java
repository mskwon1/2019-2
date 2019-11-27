package episode_1;

import java.util.*;
import java.io.*;
import java.math.*;

class Player {	
    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        int num_nodes = in.nextInt(); // the total number of nodes in the level, including the gateways
        int num_links = in.nextInt(); // the number of links
        int num_gateways = in.nextInt(); // the number of exit gateways
        
        List<Node> nodes_list = new LinkedList<Node>(); 
        
        for (int i=0;i<num_nodes;i++) {
        	nodes_list.add(new Node(i));
        }
        
        for (int i = 0; i < num_links; i++) {
            int node_a = in.nextInt(); // N1 and N2 defines a link between these nodes
            int node_b = in.nextInt();
            
            // a, b의 인접성 리스트에 각각 추가
            nodes_list.get(node_a).getAdjList().add(nodes_list.get(node_b));
            nodes_list.get(node_b).getAdjList().add(nodes_list.get(node_a));
        }
        
        ArrayList<Node> gateways = new ArrayList<Node>();
        for (int i = 0; i < num_gateways; i++) {
            int index_gateway = in.nextInt(); // the index of a gateway node
            gateways.add(nodes_list.get(index_gateway));
        }

        
        // game loop
        while (true) {
            int index_agent = in.nextInt(); // The index of the node on which the Skynet agent is positioned this turn
            
            // agent로부터 distance 입력
            BFS(nodes_list, index_agent);
            
            int min_distance = Integer.MAX_VALUE;
            List<Node> min_nodes_list = new LinkedList<Node>();

        	gateways.sort(null);
        	
            // 거리가 최소인 node들 추출
            for (Node gateway : gateways) {
            	if (gateway.getDistance() < min_distance) {
            		min_distance = gateway.getDistance();
            		min_nodes_list.clear();
            		min_nodes_list.add(gateway);
            	} else if (gateway.getDistance() == min_distance) {
            		min_nodes_list.add(gateway);
            	}
            }
            
            Node min_node = null;
            
            if (min_distance == 1) {
            	min_node = min_nodes_list.get(0);	
                
                int index_a = min_node.getIndex();
                int index_b = min_node.getParent().getIndex();
                
                min_node.getAdjList().remove(min_node.getParent());
                min_node.getParent().getAdjList().remove(min_node);
                
                System.out.println(index_a + " " + index_b);
            } else {
            	min_node = gateways.get(0);
            	boolean flag = false;
	        	for (Node gateway : gateways) {
	            	for (Node compare_gateway : gateways) {
	            		if (compare_gateway.equals(gateway)) {
	            			continue;
	            		}
	            		if (gateway.getDistance() == compare_gateway.getDistance()) {
	            			for (Node adjacent_node : gateway.getAdjList()) {
	            				if (compare_gateway.getAdjList().contains(adjacent_node)) {
	            					
	            					int index_a = gateway.getIndex();
	            					int index_b = adjacent_node.getIndex();
	            					
	            					gateway.getAdjList().remove(adjacent_node);
	            					adjacent_node.getAdjList().remove(gateway);
	            					
	            					System.out.println(index_a + " " + index_b);
	            					
	    	            			flag = true;
	    	            			break;
	            				}
	            			}
	            			if (flag) {
	            				break;
	            			}
	            		}
	            		if (flag) {
	            			break;
	            		}
	            	}
	            }
	        	if (!flag) {
	                int index_a = min_node.getIndex();
	                int index_b = min_node.getParent().getIndex();
	                
	                min_node.getAdjList().remove(min_node.getParent());
	                min_node.getParent().getAdjList().remove(min_node);
	                
	                System.out.println(index_a + " " + index_b);
	        	}
            }
            
            // Write an action using System.out.println()
            // To debug: System.err.println("Debug messages...");
            System.err.println(gateways.size());
            for (int i=0;i<gateways.size();i++) {
                System.err.println(gateways.get(i));
            }
            // Example: 0 1 are the indices of the nodes you wish to sever the link between
//            System.out.println("0 1");
        }
    }
    
	private static class Node implements Comparable<Node> {
		 int index;
		 List<Node> adjList;
		 int color; // 0 = white, 1 = gray, 2 = black
		 int distance;
		 Node parent;
	 
		 public Node(int index) {
			 this.index = index;
			 this.adjList = new LinkedList<Node>();
			 this.color = 0;
			 this.distance = -1;
			 this.parent = null;
		 }
		
		public Node getParent() {
			return parent;
		}

		public void setParent(Node parent_index) {
			this.parent = parent_index;
		}

		public int getIndex() {
			return index;
		}
	
		public int getDistance() {
			return distance;
		}

		public void setDistance(int distance) {
			this.distance = distance;
		}

		public void setIndex(int index) {
			this.index = index;
		}
	
		public List<Node> getAdjList() {
			return adjList;
		}
	
		public void setAdjList(List<Node> adjList) {
			this.adjList = adjList;
		}
	
		public int getColor() {
			return color;
		}
	
		public void setColor(int color) {
			this.color = color;
		}

		@Override
		public int compareTo(Node node) {
			if (this.distance == node.getDistance()) {
				return 0;
			} else if (this.distance > node.getDistance()){
				return 1;
			} else {
				return -1;
			}
			
		}
		
		@Override
		public String toString() {
			return "index : " + index + ", distance : " + distance;
		}
	}

	private static void BFS(List<Node> nodes_list, int index_agent) {
		Node agent_node = nodes_list.get(index_agent);
		
		for (Node node : nodes_list) {
			node.setColor(0);
			node.setDistance(Integer.MAX_VALUE);
			node.setParent(null);
		}
		
		agent_node.setColor(1);
		agent_node.setDistance(0);
		agent_node.setParent(null);
		
		Queue<Node> node_queue = new LinkedList<Node>();
		node_queue.offer(agent_node);
		
		while (!node_queue.isEmpty()) {
			Node target_node = node_queue.poll();
			for (Node child_node : target_node.getAdjList()) {
				if (child_node.getColor() == 0) {
					child_node.setColor(1);
					child_node.setDistance(target_node.getDistance()+1);
					child_node.setParent(target_node);
					node_queue.offer(child_node);
				}
			}
			target_node.setColor(2);
		}
	}
}