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
            
            // BFS 탐색을 통해 agent의 위치로부터 distance를 각각의 node를 기록
            BFS(nodes_list, index_agent, gateways);

            // gateway들을 distance 오름차순으로 정렬
        	gateways.sort(null);
        	
        	// agent로부터 제일 거리가 가까운 node를 min_node로 설정            
            Node min_node = gateways.get(0);
            int min_distance = min_node.getDistance();
            
            // 제일 가까운 노드까지의 거리가 1인 경우
            if (min_distance == 1) {      
            	// (min_node - min_node의 부모노드) 링크를 자른다
                min_node.getAdjList().remove(min_node.getParent());
                min_node.getParent().getAdjList().remove(min_node);
                
                System.out.println(min_node.getIndex() + " " + min_node.getParent().getIndex());
            } 
            // 제일 가까운 노드까지의 거리가 1보다 큰 경우
            else {
            	// 두 개 이상의 gateway와 인접한 node는 위험한 노드로 취급
            	ArrayList<Node> dangerous_nodes = getDangerousNodes(nodes_list, gateways);
            	
            	// 위험한 노드가 없는 경우
            	if (dangerous_nodes.size() == 0) {
            		// 그냥 제일 가까운 노드를 자른다
                    min_node.getAdjList().remove(min_node.getParent());
                    min_node.getParent().getAdjList().remove(min_node);
                    
                    System.out.println(min_node.getIndex() + " " + min_node.getParent().getIndex());
            	} 
            	// 위험한 노드가 있는 경우
            	else {
                	// distance 오름차순 정렬
                	dangerous_nodes.sort(null);
                	
                	// 위험한 노드들 중 우선적으로 링크를 제거해야 할 노드를 찾는다
                	Node cut_target_node = findCutTargetNode(dangerous_nodes, gateways);
                	if (cut_target_node != null) {
	                	// 찾은 노드와 근접한 gateway 간의 링크를 제거한다
	                	cutDangerousLink(cut_target_node, gateways);
                	} else {
                		// 그냥 제일 가까운 dangerous_node와 gateway간의 링크를 제거한다
                		cutDangerousLink(dangerous_nodes.get(0), gateways);
                	}
            	}
            }
        }
    }
    
    // Node 클래스
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
	
	// BFS 탐색, Gateway를 통과하는 path는 기록하지 않음 
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
					
					// 해당 노드가 gateway인 경우, 이 경로로 path를 기록하지 않음
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
	
	// 해당 dangerous node와 연결된 gateway 하나를 자름
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
	
	// gateway가 아닌 node 중 dangerous nodes를 추출 
	private static ArrayList<Node> getDangerousNodes(List<Node> nodes_list, ArrayList<Node> gateways) {
		ArrayList<Node> result_list = new ArrayList<Node>();
		for (Node node : nodes_list) {
			if (gateways.contains(node)) {
				// gateway node 인 경우 continue
				continue;
			} else {
				int count = 0;
				
				for (Node adj_node : node.getAdjList()) {
					if (gateways.contains(adj_node)) {
						// 인접한 gateway가 있으면 기록한다
						count++;
					}
				}
				
				// 2 개 이상의 인접한 gateway는 dangerous node list에 추가
				if (count > 1) {
					result_list.add(node);
				}
			}
		}
		
		return result_list;
	}

	// 우선적으로 제거해야 할 dangerous node 추출
	private static Node findCutTargetNode(ArrayList<Node> dangerous_nodes, ArrayList<Node> gateways) {
		// gateway들은 distance 순으로 오름차순 정렬 돼 있는 상태, 가까운 gateway부터 target_node가 된다
		for (Node target_node : gateways) {
			/* 
			 * target_node는 agent가 도달을 목표로 하고 있는 gateway
			 * agent가 해당 gateway로 가는 shortest path로 이동할 것이라고 가정 
			 */
        	System.err.println("PREDICTED TARGET NODE : " + target_node);
        	// Dangerous Node로 향하는 Path를 모두 점검
            for (Node dangerous_node : dangerous_nodes) {
            	System.err.print("Checking Dangerous Node : " + dangerous_node.getIndex() + "\n-----------------------\n");
            	
            	// path_node는 dangerous_node로 향하는 path에 있는 node, 자기 자신에서 시작
            	Node path_node = dangerous_node;
            	
            	// 해당 노드의 parent가 없으면 agent에 도달했다는 뜻
            	while(!(path_node.getParent() == null)) {
            		System.err.println("check path nodes : " + path_node.getIndex());
            		// path_node의 주변 node를 검사
            		for (Node adj_node : path_node.getAdjList()) {
            			if (target_node.equals(adj_node)) {
                			/*
                			 * path_node 주변에 target_node가 있으면,
                			 * agent가 target_node 쪽으로 이동 할 경우
                			 * dangerous_node 에 가까워 진다는 뜻이고
                			 * 미리 gateway들과 dangerous_node 간의 링크수를 줄여놓지 않을 경우,
                			 * 이후에 계속 agent와의 거리가 1인 Node가 있어 해당 링크를 자르는 것이 강제됐을 때
                			 * 결국 agent가 dangerous_node에 도달하면 gateway가 노출된다
                			 */
            				System.err.println("found threat : " + adj_node.getIndex());
            				
            				// 따라서 이 dangerous_node와 gateway 사이의 링크 를 지워야 한다
            				return dangerous_node;
            			}
            		}	
            		// (dangerous_node - agent) path의 다음 node로 옮겨감
            		path_node = path_node.getParent();
            	}
            	System.err.println();
            }
        	System.err.println();
    	}
		
		return null;
	}
}