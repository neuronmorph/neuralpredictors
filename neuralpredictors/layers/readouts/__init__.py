from .attention import AttentionReadout
from .base import ClonedReadout, Readout
from .factorized import FullFactorized2d, FullSXF, SpatialXFeatureLinear
from .gaussian import (
    DeterministicGaussian2d,
    FullGaussian2d,
    Gaussian2d,
    Gaussian3d,
    GeneralizedFullGaussianReadout2d,
    RemappedGaussian2d,
    UltraSparse,
    FullGaussian2d_Gumbel_softmax_scheduled_tau,
    FullGaussian2d_REINFORCE,
    FullGaussian_3d_sample_grid,
    FullGaussian2d_learnable_z_tanh,
    FullGaussian2d_Gumbel_softmax_learnable_tau,
    FullGaussian2d_Gumbel_softmax,
    FullGaussian2d_learnable_z,
    FullGaussian2d_adaptive_reg
)
from .multi_readout import MultiReadoutBase, MultiReadoutSharedParametersBase
from .point_pooled import (
    GeneralizedPointPooled2d,
    PointPooled2d,
    SpatialTransformerPooled3d,
)
from .pyramid import PointPyramid2d

### Userguide ###

# In order to build your multi-readout, pass the respective readout to your multi-readout base class
# together with your readout kwargs. Use the MultiReadoutSharedParametersBase if you want to share parameters
# between the readouts, otherwise use the MultiReadoutBase. Note that not all readouts support parameter sharing.

# Example:
# standard_multi_pointpooled_readout = MultiReadoutBase(PointPooled2d, **readout_kwargs)
# parameter_sharing_multi_gaussian_readout = MultiReadoutSharedParametersBase(FullGaussian2d, **readout_kwargs)
