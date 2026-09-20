import unittest
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
from utils.geometry_screening import pinhole_rays, candidate_geometry, geometry_decision, scannetpp_pose_in_mesh_frame, validate_scannetpp_bounds
from utils.sun_patch import PatchCandidate, filter_candidates_by_geometry

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
        self.pose = np.eye(4)
        self.scene = mi.load_dict({'type':'scene', 'panel':{'type':'rectangle',
            'to_world':mi.ScalarTransform4f().translate([0, 0, 2])}})

    def test_center_and_axis(self):
        o,d = pinhole_rays([[0,0],[1,0],[0,1]],self.K,self.pose)
        np.testing.assert_allclose(d, [[0,0,1],[2**-.5,0,2**-.5],[0,2**-.5,2**-.5]])
        np.testing.assert_allclose(o,0)

    def test_known_plane(self):
        e=candidate_geometry(np.ones((1,1),bool),self.scene,self.K,self.pose)
        self.assertEqual(e['valid_fraction'],1)
        self.assertAlmostEqual(e['median_distance'],2,places=6)
        np.testing.assert_allclose(e['median_hit_pos'],[0,0,2],atol=1e-6)
        self.assertEqual(geometry_decision(e),'surface_supported_unverified')
        self.assertEqual(geometry_decision(e,aabb_diag=1),'distant_surface')

    def test_opengl_conversion_and_miss(self):
        pose=self.pose.copy(); pose[:3,1:3]*=-1
        e=candidate_geometry(np.ones((1,1),bool),self.scene,self.K,pose)
        self.assertEqual(geometry_decision(e),'no_surface')

    def test_empty_and_invalid(self):
        e=candidate_geometry(np.zeros((1,1),bool),self.scene,self.K,self.pose)
        self.assertEqual(geometry_decision(e),'no_surface')
        with self.assertRaises(ValueError): candidate_geometry(np.ones((1,1)),self.scene,self.K,self.pose,0)
        with self.assertRaises(ValueError): pinhole_rays([[0,0]],np.zeros((3,3)),self.pose)

    def test_scannetpp_world_frame(self):
        # Independently form exporter operations on a known mesh-frame camera.
        mesh_pose=np.eye(4); mesh_pose[:3,3]=[2,3,1]
        exported=mesh_pose.copy(); exported[:3,1:3]*=-1
        exported=exported[[1,0,2,3],:]; exported[2,:]*=-1
        np.testing.assert_allclose(scannetpp_pose_in_mesh_frame(exported),mesh_pose)
        validate_scannetpp_bounds([[0,0,-3],[5,4,0]],[[0,0,0],[4,5,3]])
        with self.assertRaises(ValueError):
            validate_scannetpp_bounds([[0,0,-3],[5,4,0]],[[0,0,-3],[5,4,0]])

    def test_determinism(self):
        mask=np.ones((10,10),bool)
        a=candidate_geometry(mask,self.scene,self.K,self.pose,13,42)
        b=candidate_geometry(mask,self.scene,self.K,self.pose,13,42)
        self.assertEqual(a,b)

    def test_filter_backend_unchanged(self):
        c=PatchCandidate(np.ones((1,1),bool),np.zeros((1,1,2)),1,1,4,0,.8)
        self.assertEqual(filter_candidates_by_geometry([c],self.scene,self.K,self.pose,(1,1)),[c])
        self.assertEqual(mi.variant(),'llvm_ad_rgb')

if __name__=='__main__': unittest.main()
