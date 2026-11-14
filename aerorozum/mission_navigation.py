"""Coordinate command validation and deterministic return routing in local ENU."""
import heapq
import math


def intersects(start,end,rect,margin=.8):
    low,high=0.,1.
    for axis,key in enumerate(('x','y')):
        a,b=rect[key+'_min']-margin,rect[key+'_max']+margin
        delta=end[axis]-start[axis]
        if abs(delta)<1e-12:
            if not a<=start[axis]<=b:return False
        else:
            t0,t1=sorted(((a-start[axis])/delta,(b-start[axis])/delta))
            low,high=max(low,t0),min(high,t1)
            if low>high:return False
    return True


def validate_target(x,y,z,bounds,exclusions=(),start=None):
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in (x,y,z)):
        raise ValueError('goto coordinates and altitude must be finite numbers')
    if not bounds['x_min']<=x<=bounds['x_max'] or not bounds['y_min']<=y<=bounds['y_max'] or not 2<=z<=10:
        raise ValueError('goto is outside the flight geofence: x/y -24..24m, altitude 2..10m')
    for rect in exclusions:
        if intersects(start or (x,y),(x,y),rect):
            raise ValueError('goto path intersects an excluded region; choose an intermediate waypoint around it')
    return float(x),float(y),float(z)


def return_route(start,end,bounds,exclusions=()):
    """Shortest visible polyline around declared no-go rectangles, with clearance."""
    nodes=[tuple(start),tuple(end)]
    for r in exclusions:
        for x in (r['x_min']-1.,r['x_max']+1.):
            for y in (r['y_min']-1.,r['y_max']+1.):
                if bounds['x_min']<=x<=bounds['x_max'] and bounds['y_min']<=y<=bounds['y_max']:
                    nodes.append((x,y))
    queue=[(0.,0,[])];seen=set()
    while queue:
        cost,index,path=heapq.heappop(queue)
        if index in seen:continue
        seen.add(index)
        if index==1:return [nodes[i] for i in path+[1]]
        for target in range(len(nodes)):
            if target in seen or target==index:continue
            if not any(intersects(nodes[index],nodes[target],r) for r in exclusions):
                heapq.heappush(queue,(cost+math.dist(nodes[index],nodes[target]),target,path+([index] if index else [])))
    raise ValueError('No return path exists inside the geofence and excluded regions')
