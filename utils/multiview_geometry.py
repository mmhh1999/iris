"""Mesh-occlusion-aware projection; numpy interface, caller selects Mitsuba backend."""
import numpy as np


def intersect_rays(scene, origins, directions):
    import mitsuba as mi
    o=np.asarray(origins,dtype=float); d=np.asarray(directions,dtype=float)
    if o.shape!=d.shape or o.ndim!=2 or o.shape[1]!=3:
        raise ValueError('expected matching Nx3 rays')
    if not np.isfinite(o).all() or not np.isfinite(d).all() or np.any(np.linalg.norm(d,axis=1)==0):
        raise ValueError('invalid rays')
    d=d/np.linalg.norm(d,axis=1,keepdims=True)
    si=scene.ray_intersect(mi.Ray3f(mi.Point3f(*[mi.Float(o[:,i]) for i in range(3)]),mi.Vector3f(*[mi.Float(d[:,i]) for i in range(3)])))
    return np.column_stack([np.asarray(si.p[i]) for i in range(3)]), np.asarray(si.is_valid()),np.asarray(si.t)


def project_points(points, K, c2w, img_hw):
    p=np.asarray(points,dtype=float); pose=np.asarray(c2w,dtype=float)
    cam=(p-pose[:3,3])@pose[:3,:3]
    front=cam[:,2]>1e-8
    xy=np.full((len(p),2),np.nan)
    homogeneous=cam[front]@np.asarray(K).T
    xy[front]=homogeneous[:,:2]/homogeneous[:,2,None]-.5
    h,w=img_hw
    inside=front & (xy[:,0]>=0) & (xy[:,0]<=w-1) & (xy[:,1]>=0) & (xy[:,1]<=h-1)
    return xy,inside


def visible_projections(points, scene, K, c2w, img_hw, tolerance=.04):
    if tolerance<=0: raise ValueError('positive visibility tolerance required')
    xy,inside=project_points(points,K,c2w,img_hw)
    ids=np.flatnonzero(inside); visible=np.zeros(len(points),bool)
    if len(ids):
        origin=np.broadcast_to(np.asarray(c2w)[:3,3],(len(ids),3))
        delta=np.asarray(points)[ids]-origin
        _,valid,t=intersect_rays(scene,origin,delta)
        visible[ids]=valid & (np.abs(t-np.linalg.norm(delta,axis=1))<=tolerance)
    return xy,visible


def intersect_plane(origins,directions,point,normal):
    o=np.asarray(origins);d=np.asarray(directions);n=np.asarray(normal)
    denom=d@n
    with np.errstate(divide='ignore',invalid='ignore'):
        t=((np.asarray(point)-o)@n)/denom
    valid=(np.abs(denom)>1e-10)&(t>0)&np.isfinite(t)
    pts=np.full(o.shape,np.nan);pts[valid]=o[valid]+t[valid,None]*d[valid]
    return pts,valid


def aperture_distances(points, directions, aperture):
    """Ray distances to a vertical rectangular portal; inf for no forward hit.

    points/directions must broadcast to (...,3); direction need not be unit.
    """
    p=np.asarray(points);d=np.asarray(directions)
    a,b=np.asarray(aperture['endpoints_xy'],dtype=float)
    tangent=b-a; length=np.linalg.norm(tangent)
    z0,z1=aperture['z_range']
    if length<=0 or z1<=z0: raise ValueError('degenerate aperture')
    tangent=tangent/length; normal=np.array([-tangent[1],tangent[0],0])
    anchor=np.array([a[0],a[1],z0])
    denom=d@normal
    with np.errstate(divide='ignore',invalid='ignore'):
        t=((anchor-p)@normal)/denom
        hit=p+t[...,None]*d
    along=(hit[...,:2]-a)@tangent
    valid=(np.abs(denom)>1e-9)&(t>0)&(along>=0)&(along<=length)&(hit[...,2]>=z0)&(hit[...,2]<=z1)
    return np.where(valid,t,np.inf)


def sunlight_visibility(points, directions, scene, apertures, tolerance=.04):
    """N receivers x M sun directions: portal hit before any opaque mesh hit.

    Four-centimeter default tolerance admits imperfect mesh near glazing; this
    is a hypothesis requiring sensitivity analysis, not a transmission model.
    """
    p=np.asarray(points);d=np.asarray(directions,dtype=float)
    d=d/np.linalg.norm(d,axis=1,keepdims=True)
    result=np.zeros((len(d),len(p)),bool)
    # Chunk to bound memory while preserving deterministic ray order.
    for begin in range(0,len(d),32):
        ds=d[begin:begin+32]
        origins=np.broadcast_to(p,(len(ds),len(p),3)).copy()
        rays=np.broadcast_to(ds[:,None,:],origins.shape).copy()
        portal=np.full(origins.shape[:2],np.inf)
        for aperture in apertures:
            portal=np.minimum(portal,aperture_distances(origins,rays,aperture))
        active=np.isfinite(portal)
        if not active.any():continue
        oo=origins[active]+.002*rays[active]
        _,valid,t=intersect_rays(scene,oo,rays[active])
        lit=np.zeros(active.shape,bool)
        lit[active]=(~valid)|(t+.002>=portal[active]-tolerance)
        result[begin:begin+len(ds)]=lit
    return result
