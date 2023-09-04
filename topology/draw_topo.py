#coding=utf-8
import networkx as nx
import matplotlib.pyplot as plt
import sys

file_name = sys.argv[1]
topo_file = open(file_name, "r")
content = topo_file.readline()

n, m = list(map(int, content.split()))
linkSet = []
G = nx.Graph()
for i in range(m):
    content = topo_file.readline()
    u, v, w, c, _ = list(map(int, content.split()))
    G.add_edge(u, v, weight=c)
#G = nx.Graph(linkSet, edge_attr='weight')
print("degree:", G.degree())
nx.draw(G, with_labels=True, width=[float(d['weight']/5000) for (u,v,d) in G.edges(data=True)])
plt.show()
#plt.savefig("./figures/GEA_topo.eps")
