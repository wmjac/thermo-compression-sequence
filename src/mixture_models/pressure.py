"""Thermodynamic total pressure from a model's excess density."""


def total_pressure(model, inputs):
    out = model(inputs)
    return inputs["ctot"] + out["pressure_excess"]
