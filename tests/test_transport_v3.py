"""Physical invariants for full-house diffuse transport and finite lights."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from nucleus4d_mesh_bake import setup
from nucleus4d_transport_bake import direct_light,radiosity,sun_state
from nucleus4d_transport_sky_mis import evaluate as sky_mis
from nucleus4d_transport_emitters import SourceScene


class TransportV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.mi,cls.dr=setup(2)

    def test_finite_downlight_matches_exact_on_axis_disk_irradiance(self):
        mi=self.mi;sc=mi.load_dict({'type':'scene'})
        light=dict(id=0,type='disk',center=[0,0,1],radius=.05,radiance=[1,1,1])
        # Lambertian receiver parallel to a coaxial disc: E/pi=L*R²/(R²+h²).
        for height,tolerance in [(.01,.003),(.15,.003),(2.,1e-5)]:
            p=np.array([[0,0,1-height]],np.float32);n=np.array([[0,0,1]],np.float32)
            value=direct_light(sc,p,n,light)
            expected=.05**2/(.05**2+height**2)
            np.testing.assert_allclose(value,expected,atol=tolerance)

    def test_opaque_intervening_wall_blocks_actual_finite_source(self):
        mi=self.mi;sc=mi.load_dict({'type':'scene','block':{'type':'rectangle','to_world':mi.ScalarTransform4f.translate([0,0,.5]).scale(10)}})
        light=dict(id=0,type='disk',center=[0,0,1],radius=.05,radiance=[1,1,1])
        np.testing.assert_array_equal(direct_light(sc,np.array([[0,0,0]],np.float32),np.array([[0,0,1]],np.float32),light),0)

    def test_independent_lights_superpose_and_zero_power_is_zero_energy(self):
        F=sparse.csr_matrix(np.array([[0,.4],[.4,0]],np.float32))
        rho=np.array([[.5,.4,.3]],np.float32)
        a=np.array([[1,.5,.2],[0,0,0]],np.float32);b=np.array([[0,0,0],[.2,.4,.8]],np.float32)
        np.testing.assert_allclose(radiosity(F,rho,a+b),radiosity(F,rho,a)+radiosity(F,rho,b),rtol=1e-6)
        np.testing.assert_array_equal(radiosity(F,rho,np.zeros_like(a)),0)
        self.assertTrue((radiosity(F,rho,a)[1]>0).all())

    def test_solar_direction_is_one_moving_unit_vector(self):
        a,_=sun_state(750);b,_=sun_state(1095);middle,_=sun_state(920)
        self.assertGreater(float(np.linalg.norm(a-b)),.5)
        self.assertGreater(float(np.linalg.norm(a-middle)),.2)
        for minute in np.linspace(750,1095,100):
            d,p=sun_state(minute);self.assertAlmostEqual(float(np.linalg.norm(d)),1,places=12)
            self.assertTrue((p>=0).all())

    def test_window_mis_preserves_constant_sky_including_portal_plane(self):
        mi=self.mi;sc=mi.load_dict({'type':'scene'});environment=mi.load_dict({'type':'constant','radiance':{'type':'rgb','value':1.}})
        p=np.array([[0,0,-1],[3,-1,0],[6,0,0]],np.float32);n=np.tile([0,0,1.],(len(p),1)).astype(np.float32)
        value=sky_mis(sc,[environment,environment],p,n,4096)
        np.testing.assert_allclose(value,1,atol=.005)

    def test_sampling_portals_do_not_leak_through_a_closed_room(self):
        mi=self.mi;sc=mi.load_dict({'type':'scene','room':{'type':'cube'}})
        environment=mi.load_dict({'type':'constant','radiance':{'type':'rgb','value':1.}})
        value=sky_mis(sc,[environment,environment],np.array([[0,0,0]],np.float32),np.array([[0,0,1]],np.float32),1024)
        np.testing.assert_array_equal(value,0)

    def test_lamp_housing_alignment_keeps_intervening_walls_opaque(self):
        mi=self.mi;light=dict(id=0,type='disk',center=[0,0,1],radius=.05,radiance=[1,1,1])
        housing={'type':'rectangle','to_world':mi.ScalarTransform4f.translate([0,0,.97]).scale(.1)}
        p=np.array([[0,0,0]],np.float32);n=np.array([[0,0,1]],np.float32)
        sc=mi.load_dict({'type':'scene','housing':housing})
        np.testing.assert_array_equal(direct_light(sc,p,n,light),0)
        np.testing.assert_allclose(direct_light(SourceScene(sc,light),p,n,light),.05**2/(1+.05**2),atol=1e-5)
        blocked=mi.load_dict({'type':'scene','housing':housing,'wall':{'type':'rectangle','to_world':mi.ScalarTransform4f.translate([0,0,.5]).scale(1)}})
        np.testing.assert_array_equal(direct_light(SourceScene(blocked,light),p,n,light),0)


if __name__=='__main__':unittest.main()
