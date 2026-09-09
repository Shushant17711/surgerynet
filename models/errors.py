class ShapeInfeasibleError(RuntimeError):
    """Raised by a fixed-input decoder (MLP, CNN) when given an input whose
    size/shape doesn't match what it was built/trained for.

    This is the point of design doc §5.4: MLP and CNN decoders cannot even
    be *evaluated* on lattice surgery's merge window, because the input
    isn't just harder there, it's a different shape (a merged patch is
    bigger than either standalone patch). Experiment scripts (E3) catch
    this specific exception to report those rows as "N/A — architecturally
    infeasible" rather than letting them crash or silently mis-decode.
    """
