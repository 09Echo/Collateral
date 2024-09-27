from transformers import PretrainedConfig

class MambaVisionConfig(PretrainedConfig):
    model_type = "mambavision"

    def __init__(
        self,
        depths=[1, 3, 8, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs,
    ):
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dim = dim
        self.in_dim = in_dim
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        super().__init__(**kwargs)