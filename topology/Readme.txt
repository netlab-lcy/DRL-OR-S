The topology file directory includes:
1. Directory for topology structure information, e.g., Abi.txt
2. Topology traffic matrix in one line, an n by n matrix where the element in the i-th row and j-th column represents the traffic from node i to node j.

Topology File Structure
The first line contains n and m, representing the number of nodes and links respectively. Links are bidirectional.
The next m lines each represent parameters for a link:
u, v (numbered from 1) w (link weight) c (link bandwidth) loss (link packet loss rate in percentage)
Following this are n binary variables representing whether a node is controlled by a DRL agent.

