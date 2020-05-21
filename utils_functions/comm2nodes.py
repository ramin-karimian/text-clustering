
from collections import defaultdict


def write_comm2nodes(conf,filename,th,delimiter="\t"):
    """writes the .edge2comm, .comm2edges, and .comm2nodes files"""
    comm2nodes_filepath = f"{conf['resultspath']}/{filename[:-6]}_{th}.comm2nodes"
    clusters_filepath = f"{conf['resultspath']}/{filename[:-6]}_{th}.clusters"

    f = open(clusters_filepath,'r')
    lns = f.readlines()
    lns = [l[:-2].split(" ") for l in lns ]
    f.close()
    cid2edges,cid2nodes = defaultdict(set),defaultdict(set) # faster to recreate here than

    for i,cl in enumerate(lns):
        for e in cl:
            nodes = e.split(",")
            cid2nodes[i] |= set(nodes)
    cid2nodes = dict(cid2nodes)

    # write list of edges for each comm, each comm on its own line
    # first entry of each line is cid
    g = open(comm2nodes_filepath, 'w')
    for cid in sorted(cid2nodes.keys()):
        strcid = str(cid)
        nodes  = map(str,cid2nodes[cid])
        g.write( delimiter.join([strcid] + list(nodes)) )
        g.write( "\n" )
    g.close()
