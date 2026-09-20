import unittest
import numpy as np
import mitsuba as mi
from experiments.evaluate_scannetpp_rerender import visibility_batch


class RerenderVisibilityTests(unittest.TestCase):
    def setUp(self):
        mi.set_variant('llvm_ad_rgb')
        self.scene=mi.load_dict({'type':'scene','occluder':{
            'type':'rectangle','to_world':mi.ScalarTransform4f().translate([0,0,1])}})

    def test_known_occlusion_and_open_sky(self):
        points=np.array([[0.,0.,0.],[2.,0.,0.]])
        directions=np.array([[0.,0.,1.],[0.,0.,-1.],[.2,0.,1.]])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        actual=visibility_batch(self.scene,points,directions,.002)
        np.testing.assert_array_equal(actual,[[False,True],[True,True],[False,True]])

    def test_chunk_boundary_keeps_direction_order(self):
        points=np.array([[0.,0.,0.],[2.,0.,0.]])
        directions=np.array([[0.,0.,1.] if i%2==0 else [0.,0.,-1.] for i in range(67)])
        actual=visibility_batch(self.scene,points,directions,.002)
        np.testing.assert_array_equal(actual,[[i%2==1,True] for i in range(67)])


if __name__=='__main__':unittest.main()
