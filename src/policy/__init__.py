from src.policy.baseline.bc_baseline import Simple_driver
__all__ = {
    'baseline': Simple_driver,
}


def build_model(config):
    model = __all__[config.method.model_name](**config.method)
    return model