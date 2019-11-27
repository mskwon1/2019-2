package programmers;

import java.util.Scanner;
import java.util.List;
import java.util.Queue;
import java.util.ArrayList;
import java.util.LinkedList;

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
            
            // a, b�� ������ ����Ʈ�� ���� �߰�
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
            
            // BFS Ž���� ���� agent�� ��ġ�κ��� distance�� ������ node�� ���
            BFS(nodes_list, index_agent, gateways);

            // gateway���� distance ������������ ����
        	gateways.sort(null);
        	
        	// agent�κ��� ���� �Ÿ��� ����� node�� min_node�� ����            
            Node min_node = gateways.get(0);
            int min_distance = min_node.getDistance();
            
            // ���� ����� �������� �Ÿ��� 1�� ���
            if (min_distance == 1) {      
            	// (min_node - min_node�� �θ���) ��ũ�� �ڸ���
                min_node.getAdjList().remove(min_node.getParent());
                min_node.getParent().getAdjList().remove(min_node);
                
                System.out.println(min_node.getIndex() + " " + min_node.getParent().getIndex());
            } 
            // ���� ����� �������� �Ÿ��� 1���� ū ���
            else {
            	// �� �� �̻��� gateway�� ������ node�� ������ ���� ���
            	ArrayList<Node> dangerous_nodes = getDangerousNodes(nodes_list, gateways);
            	
            	// ������ ��尡 ���� ���
            	if (dangerous_nodes.size() == 0) {
            		// �׳� ���� ����� ��带 �ڸ���
                    min_node.getAdjList().remove(min_node.getParent());
                    min_node.getParent().getAdjList().remove(min_node);
                    
                    System.out.println(min_node.getIndex() + " " + min_node.getParent().getIndex());
            	} 
            	// ������ ��尡 �ִ� ���
            	else {
                	// distance �������� ����
                	dangerous_nodes.sort(null);
                	
                	// ������ ���� �� �켱������ ��ũ�� �����ؾ� �� ��带 ã�´�
                	Node cut_target_node = findCutTargetNode(dangerous_nodes, gateways);
                	if (cut_target_node != null) {
	                	// ã�� ���� ������ gateway ���� ��ũ�� �����Ѵ�
	                	cutDangerousLink(cut_target_node, gateways);
                	} else {
                		// �׳� ���� ����� dangerous_node�� gateway���� ��ũ�� �����Ѵ�
                		cutDangerousLink(dangerous_nodes.get(0), gateways);
                	}
            	}
            }
        }
    }
    
    // Node Ŭ����
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
	
	// BFS Ž��, Gateway�� ����ϴ� path�� ������� ���� 
	private static void BFS(List<Node> nodes_list, int index_agent, ArrayList<Node> gateways) {
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
					
					// �ش� ��尡 gateway�� ���, �� ��η� path�� ������� ����
					if (gateways.contains(child_node)) {
						continue;
					} else {
						node_queue.offer(child_node);
					}
				}
			}
			target_node.setColor(2);
		}
	}
	
	// �ش� dangerous node�� ����� gateway �ϳ��� �ڸ�
	private static void cutDangerousLink(Node dangerous_node, ArrayList<Node> gateways) {
		for (Node adj_node : dangerous_node.getAdjList()) {
			if (gateways.contains(adj_node)) {
				dangerous_node.getAdjList().remove(adj_node);
				adj_node.getAdjList().remove(dangerous_node);
				
				System.out.println(dangerous_node.getIndex() + " " + adj_node.getIndex());
				
				return;
			}
		}
	}
	
	// gateway�� �ƴ� node �� dangerous nodes�� ���� 
	private static ArrayList<Node> getDangerousNodes(List<Node> nodes_list, ArrayList<Node> gateways) {
		ArrayList<Node> result_list = new ArrayList<Node>();
		for (Node node : nodes_list) {
			if (gateways.contains(node)) {
				// gateway node �� ��� continue
				continue;
			} else {
				int count = 0;
				
				for (Node adj_node : node.getAdjList()) {
					if (gateways.contains(adj_node)) {
						// ������ gateway�� ������ ����Ѵ�
						count++;
					}
				}
				
				// 2 �� �̻��� ������ gateway�� dangerous node list�� �߰�
				if (count > 1) {
					result_list.add(node);
				}
			}
		}
		
		return result_list;
	}

	// �켱������ �����ؾ� �� dangerous node ����
	private static Node findCutTargetNode(ArrayList<Node> dangerous_nodes, ArrayList<Node> gateways) {
		// gateway���� distance ������ �������� ���� �� �ִ� ����, ����� gateway���� target_node�� �ȴ�
		for (Node target_node : gateways) {
			/* 
			 * target_node�� agent�� ������ ��ǥ�� �ϰ� �ִ� gateway
			 * agent�� �ش� gateway�� ���� shortest path�� �̵��� ���̶�� ���� 
			 */
        	System.err.println("PREDICTED TARGET NODE : " + target_node);
        	// Dangerous Node�� ���ϴ� Path�� ��� ����
            for (Node dangerous_node : dangerous_nodes) {
            	System.err.print("Checking Dangerous Node : " + dangerous_node.getIndex() + "\n-----------------------\n");
            	
            	// path_node�� dangerous_node�� ���ϴ� path�� �ִ� node, �ڱ� �ڽſ��� ����
            	Node path_node = dangerous_node;
            	
            	// �ش� ����� parent�� ������ agent�� �����ߴٴ� ��
            	while(!(path_node.getParent() == null)) {
            		System.err.println("check path nodes : " + path_node.getIndex());
            		// path_node�� �ֺ� node�� �˻�
            		for (Node adj_node : path_node.getAdjList()) {
            			if (target_node.equals(adj_node)) {
                			/*
                			 * path_node �ֺ��� target_node�� ������,
                			 * agent�� target_node ������ �̵� �� ���
                			 * dangerous_node �� ����� ���ٴ� ���̰�
                			 * �̸� gateway��� dangerous_node ���� ��ũ���� �ٿ����� ���� ���,
                			 * ���Ŀ� ��� agent���� �Ÿ��� 1�� Node�� �־� �ش� ��ũ�� �ڸ��� ���� �������� ��
                			 * �ᱹ agent�� dangerous_node�� �����ϸ� gateway�� ����ȴ�
                			 */
            				System.err.println("found threat : " + adj_node.getIndex());
            				
            				// ���� �� dangerous_node�� gateway ������ ��ũ �� ������ �Ѵ�
            				return dangerous_node;
            			}
            		}	
            		// (dangerous_node - agent) path�� ���� node�� �Űܰ�
            		path_node = path_node.getParent();
            	}
            	System.err.println();
            }
        	System.err.println();
    	}
		
		return null;
	}
}