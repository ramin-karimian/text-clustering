from nf1 import NF1
import networkx as nx
from networkx.algorithms import community

g = nx.karate_club_graph()

kclique = list(community.k_clique_communities(g, 4))
kcoms = [tuple(x) for x in kclique]

lp = list(community.label_propagation_communities(g))
lpcoms = [tuple(x) for x in lp]

# Computing the NF1 scores and statistics
nf = NF1(lpcoms, kcoms)
results = nf.summary()
print(results['scores'])
print(results['details'])
print(results['scores'].loc['NF1'])

# Visualising the Precision-Recall density scatter-plot
# nf.plot()
