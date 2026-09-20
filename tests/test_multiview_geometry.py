import unittest
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
from utils.geometry_screening import pinhole_rays
from utils.multiview_geometry import project_points,visible_projections,intersect_plane

class MultiViewTests(unittest.TestCase):
    def test_roundtrip(self):
        K=np.array([[20.,0,10.],[0,20.,10.],[0,0,1.]])
        pose=np.eye(4);pose[:3,3]=[1,2,3]
        pixels=np.array([[0.,0.],[4.,5.],[19.,19.]])
        o,d=pinhole_rays(pixels,K,pose)
        xy,valid=project_points(o+3*d,K,pose,(20,20))
        np.testing.assert_allclose(xy,pixels,atol=1e-12)
        self.assertTrue(valid[1])

    def test_behind_camera(self):
        xy,valid=project_points([[0,0,-1]],np.eye(3),np.eye(4),(10,10))
        self.assertFalse(valid[0]);self.assertTrue(np.isnan(xy).all())

    def test_occluded(self):
        scene=mi.load_dict({'type':'scene','r':{'type':'rectangle','to_world':mi.ScalarTransform4f().translate([0,0,2])}})
        K=np.array([[1.,0,.5],[0,1.,.5],[0,0,1.]])
        _,v=visible_projections(np.array([[0,0,2],[0,0,3]]),scene,K,np.eye(4),(1,1))
        np.testing.assert_equal(v,[True,False])

    def test_plane(self):
        p,v=intersect_plane(np.array([[0,0,1]]*3),np.array([[0,0,-1],[1,0,0],[0,0,1]]),[0,0,0],[0,0,1])
        np.testing.assert_equal(v,[True,False,False]);np.testing.assert_allclose(p[0],0)

if __name__=='__main__':unittest.main()

class PortalTests(unittest.TestCase):
    def test_portal_forward_and_backward(self):
        from utils.multiview_geometry import aperture_distances
        aperture={'endpoints_xy':[[-1,0],[1,0]],'z_range':[1,2]}
        p=np.array([[0,2,0]]*3); d=np.array([[0,-1,.75],[0,1,.75],[2,-1,.75]])
        t=aperture_distances(p,d,aperture)
        self.assertAlmostEqual(t[0],2);self.assertTrue(np.isinf(t[1:]).all())

    def test_portal_shadow(self):
        from utils.multiview_geometry import sunlight_visibility
        aperture={'endpoints_xy':[[-2,2],[2,2]],'z_range':[0,3]}
        empty=mi.load_dict({'type':'scene'})
        blocked=mi.load_dict({'type':'scene','s':{'type':'sphere','center':[0,1,1],'radius':.3}})
        p=np.array([[0,0,0.]])
        np.testing.assert_equal(sunlight_visibility(p,[[0,1,1]],empty,[aperture]),[[True]])
        np.testing.assert_equal(sunlight_visibility(p,[[0,1,1]],blocked,[aperture]),[[False]])
