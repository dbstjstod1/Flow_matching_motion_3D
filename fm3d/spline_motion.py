"""Spline/optimizer schemes for the common FM loop, without changing its operators.

The Akima formula is vectorized over six components and gated against Thies' release.
The cubic B-spline basis is exported from JRM-ADM's interpolation package on CPU.
Spline coefficients use mm/degrees; all loop poses use mm/axis-angle radians.
These are adapted motion estimation modules, not reproductions of complete baselines.
"""
from pathlib import Path
import math
import numpy as np
import torch
from .motion_estimation import _BaseEstimator

BASIS = Path(__file__).resolve().parent / 'assets/jrm_cubic_bspline_V360_C20.npy'


class AkimaInterpolator:
    def __init__(self, views, nodes, device, dtype=torch.float32):
        self.nodes = nodes
        t = torch.linspace(0, views-1, nodes, device=device, dtype=dtype)
        q = torch.arange(views, device=device, dtype=dtype)
        self.dx = t.diff()[:, None]
        self.idx = (torch.searchsorted(t, q, right=True)-1).clamp(0,nodes-2)
        self.offset = (q-t[self.idx])[:, None]

    def __call__(self, values):
        n = self.nodes
        m = values.diff(dim=0)/self.dx
        left, right = 2*m[:1]-m[1:2], 2*m[-1:]-m[-2:-1]
        slopes = torch.cat([2*left-m[:1], left, m, right, 2*right-m[-1:]])
        dm = slopes.diff(dim=0).abs()
        f1, f2 = dm[2:n+2], dm[:n]
        total = f1+f2
        valid = total > 1e-8*total.amax(dim=0,keepdim=True)
        denominator = torch.where(valid,total,torch.ones_like(total))
        b = torch.where(valid,(f1*slopes[1:n+1]+f2*slopes[2:n+2])/denominator,slopes[1:n+1])
        c = (3*m-2*b[:-1]-b[1:])/self.dx
        d = (b[:-1]+b[1:]-2*m)/self.dx.square()
        i,z = self.idx,self.offset
        return values[i]+b[i]*z+c[i]*z.square()+d[i]*z.pow(3)


class SplineSchemeEstimator(_BaseEstimator):
    def __init__(self,*args,scheme,lr,**kwargs):
        kwargs.pop('opt',None)
        kwargs.pop('smooth_w',None)
        super().__init__(*args,smooth_w=0.0,**kwargs)
        self.scheme=scheme
        self.lr0=lr
        self.calls=0
        if scheme=='akima_gd':
            self.n_ctrl=30
            self.interpolator=AkimaInterpolator(self.V,30,self.device)
            self.decay,self.decay_reset=.97,'call'
        elif scheme=='bspline_rmsprop':
            self.n_ctrl=20
            self.B=torch.tensor(np.load(BASIS, allow_pickle=False),device=self.device)
            if self.B.shape!=(self.V,20): raise ValueError('B-spline asset requires 360 views')
        else: raise ValueError(scheme)
        self.coefficients=torch.zeros(self.n_ctrl,6,device=self.device,requires_grad=True)
        self.units=self.coefficients.new_tensor([1,1,1,math.pi/180,math.pi/180,math.pi/180])
        self._optim=self.new_optimizer()

    def new_optimizer(self):
        if self.scheme=='akima_gd':
            return torch.optim.SGD([self.coefficients],lr=self.lr0)
        return torch.optim.RMSprop([self.coefficients],lr=self.lr0,alpha=.99,eps=1e-8,
                                   momentum=0,centered=False)

    def _theta(self):
        values=self.interpolator(self.coefficients) if self.scheme=='akima_gd' else self.B@self.coefficients
        return values*self.units

    def refine_global(self,*args,**kwargs):
        # Preserve fitted spline coefficients; released JRM resets RMSprop moments
        # between calls. GD resets its exponential schedule within each fit.
        if self.scheme=='bspline_rmsprop': self._optim=self.new_optimizer()
        loss=super().refine_global(*args,**kwargs)
        if not math.isfinite(loss) or not torch.isfinite(self.coefficients).all():
            raise FloatingPointError(f'Nonfinite optimization in {self.scheme}')
        self.calls+=1
        return loss
