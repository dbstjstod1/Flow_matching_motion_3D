"""CPU numerical gates for spline values, gradients, units and update continuity."""
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bench.thies.motion import akima_resample
from fm3d.spline_motion import AkimaInterpolator,SplineSchemeEstimator


def main():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    for zero in [False,True]:
        x=(torch.zeros(30,6,dtype=torch.float64) if zero else torch.randn(30,6,dtype=torch.float64)).requires_grad_()
        a=AkimaInterpolator(360,30,'cpu',torch.float64)(x)
        b=akima_resample(x,360)
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
        weights=torch.randn_like(a)
        ga=torch.autograd.grad((a*weights).sum(),x,retain_graph=True)[0]
        gb=torch.autograd.grad((b*weights).sum(),x)[0]
        torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-10)
    cfg=SimpleNamespace(n_views=360,du=1.)
    def make(scheme):
        e=SplineSchemeEstimator(cfg,None,None,None,'cpu',scheme=scheme,
            lr=5. if scheme=='akima_gd' else .02,loss='l2',views_per_iter=24)
        def project(self,image,theta,views=None):
            values=theta/self.units
            return (values if views is None else values[views]).reshape(-1,2,3)
        e._project=MethodType(project,e)
        return e
    for scheme in ['akima_gd','bspline_rmsprop']:
        a,b=make(scheme),make(scheme)
        assert torch.count_nonzero(a.current_params())==0 and a.smooth_w==0
        target=torch.ones(360,2,3)*.4
        for _ in range(3):
            a.refine_global(None,target,80)
            b.refine_global(None,target,80)
        assert (a._project(None,a.current_params())-target).square().mean()<.16
        torch.testing.assert_close(a.current_params(),b.current_params(),atol=0,rtol=0)
        with torch.no_grad():a.coefficients.fill_(1.)
        torch.testing.assert_close(a.current_params(),a.units.expand(360,6),atol=1e-5,rtol=1e-5)
        if scheme=='akima_gd': assert abs(a._optim.param_groups[0]['lr']-5*.97**79)<1e-12
    print('PASS: Akima released values/gradients including zero initialization; spline units, convergence, deterministic view sampling and update scheduling.')


if __name__=='__main__':main()
