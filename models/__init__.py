from .simple_3d import Simple3DConvNet
from .simple_2d import Simple2DConvNet
from .fusion import PerFrameAggregator, EarlyFusion2D, LateFusionEnsemble
from .dual_stream import TwoStreamConvNet


def create_model(name: str, num_classes: int, n_frames: int = 10, fusion_agg: str = "mean"):
    name = name.lower()
    if name == "3d":
        return Simple3DConvNet(in_ch=3, num_classes=num_classes)
    elif name == "2d_per_frame_avg":
        frame_net = Simple2DConvNet(in_ch=3, num_classes=num_classes)
        return PerFrameAggregator(frame_model=frame_net, agg=fusion_agg)
    elif name == "early_fusion_2d":
        # underlying 2D model expects C*T input channels
        model_2d = Simple2DConvNet(in_ch=3 * n_frames, num_classes=num_classes)
        return EarlyFusion2D(model_2d=model_2d, n_frames=n_frames)
    elif name == "late_fusion":
        # simple example: fuse two different 3D submodels
        sub1 = Simple3DConvNet(in_ch=3, num_classes=num_classes)
        sub2 = Simple3DConvNet(in_ch=3, num_classes=num_classes)
        return LateFusionEnsemble([sub1, sub2], agg=fusion_agg)
    elif name in {"two_stream", "dual_stream"}:
        if fusion_agg == "mean":
            fusion = "mean"
        elif fusion_agg == "max":
            fusion = "max"
        else:
            fusion = "sum"
        return TwoStreamConvNet(
            num_classes=num_classes,
            n_frames=n_frames,
            fusion=fusion,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
