# Copyright (c) 2018 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by the AGPLv3 license that can be
# found in the LICENSE file.

def find_ccs(hot_spots):
    ''' find connected components.
        hot_spots is a list of (x, y) locations that are "hot"
        returns list of (x, y) points corresponding to the mean location of each CC.
    '''
    def uf_root(id):
        while ids[id] != id:
            ids[id] = ids[ids[id]]
            id = ids[id]
        return id

    def uf_unite(id1, id2):
        id1_root = uf_root(id1)
        id2_root = uf_root(id2)
        if id1_root != id2_root:
            if szs[id1_root] < szs[id2_root]:
                ids[id1_root] = id2_root
                szs[id2_root] += szs[id1_root]
            else:
                ids[id2_root] = id1_root
                szs[id1_root] += szs[id2_root]
    
    ids = [i for i in range(len(hot_spots))]
    szs = [1 for _ in range(len(hot_spots))]

    for i, (xi, yi) in enumerate(hot_spots):
        for j, (xj, yj) in enumerate(hot_spots):
            if i != j and abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
                uf_unite(i, j)
    
    regions = [[] for _ in range(len(hot_spots))]
    for i, (x, y) in enumerate(hot_spots):
        root = uf_root(i)
        regions[root].append((x, y))
    
    return [(sum(zip(*r)[0])/len(r), sum(zip(*r)[1])/len(r)) for r in regions if len(r) > 0]
