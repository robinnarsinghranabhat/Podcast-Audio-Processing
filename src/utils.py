def norm_spec(spec):
    return (spec - spec.min()) / (spec.max() - spec.min())
