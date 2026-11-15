"""Versioned Aki-box containment score with a configurable edge allowance."""

def area(b):return max(0.,b['x_max']-b['x_min'])*max(0.,b['y_max']-b['y_min'])

def iou(a,b):
    w=max(0.,min(a['x_max'],b['x_max'])-max(a['x_min'],b['x_min']));h=max(0.,min(a['y_max'],b['y_max'])-max(a['y_min'],b['y_min']));inter=w*h
    return inter/max(1e-12,area(a)+area(b)-inter)

def edge_close(pred,gold,fraction=.75):
    """Every predicted edge stays within the gold box expanded by fraction."""
    tx=(gold['x_max']-gold['x_min'])*fraction;ty=(gold['y_max']-gold['y_min'])*fraction
    return pred['x_min']>=gold['x_min']-tx and pred['x_max']<=gold['x_max']+tx and pred['y_min']>=gold['y_min']-ty and pred['y_max']<=gold['y_max']+ty

def score(predicted,gold,iou_threshold=.5,edge_tolerance=.75):
    predicted=[b.get('bbox',b) for b in predicted];gold=[b.get('bbox',b) for b in gold]
    edges={i:[j for j,g in enumerate(gold) if iou(p,g)>=iou_threshold or edge_close(p,g,edge_tolerance)] for i,p in enumerate(predicted)};matched={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in matched or augment(matched[j],seen):matched[j]=i;return True
        return False
    for i in range(len(predicted)):augment(i,set())
    tp=len(matched);fp=len(predicted)-tp;fn=len(gold)-tp;f2=5*tp/max(1,5*tp+4*fn+fp)
    return dict(tp=tp,fp=fp,fn=fn,f2=f2,iou_threshold=iou_threshold,edge_tolerance_fraction=edge_tolerance)
