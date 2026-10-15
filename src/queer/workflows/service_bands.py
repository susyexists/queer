"""Native imported-band and calculation postprocessing adapters."""

from dataclasses import replace

import numpy as np

from ..observables.result import ObservableResult
from .service_sources import _postprocess_inputs


def _band_observable(bands, *, summary=None, source=None):
    result = bands.as_result()
    return ObservableResult(
        result.arrays,
        result.dims,
        result.units,
        metadata={
            **result.metadata,
            "labels": list(bands.labels),
            "label_indices": list(map(int, bands.label_indices)),
            "segments": list(bands.segments),
            "summary": summary or {},
            "source": source,
        },
    )


def _postprocess(config):
    from ..io import bands as readers
    from ..io import calculations

    parameters = config.parameters
    engine = parameters.get("engine", "vasp")
    available = _postprocess_inputs(config)
    source = available.get("source")
    summary, banddata, arrays, dims, units = {}, None, {}, {}, {}

    def consume(parsed):
        nonlocal banddata
        result = parsed.as_result()
        arrays.update(result.arrays)
        dims.update(result.dims)
        units.update(result.units)
        summary.update(parsed.summary)
        if parsed.bands is not None:
            banddata = parsed.bands

    if engine == "vasp":
        if "outcar" in available or (source and source.name.upper() == "OUTCAR"):
            consume(calculations.read_vasp_results(available.get("outcar", source)))
        if "vasprun" in available or (source and source.suffix == ".xml"):
            consume(calculations.read_vasprun_results(available.get("vasprun", source)))
        if banddata is None and ("eigenval" in available or (source and source.name.upper() == "EIGENVAL")):
            banddata = readers.read_eigenval(
                available.get("eigenval", source), fermi_energy=parameters.get("fermi_energy")
            )
    elif engine == "openmx":
        if "output" in available or (source and source.suffix.lower() in {".out", ".log"}):
            consume(calculations.read_openmx_results(available.get("output", source)))
        if "bandfile" in available or (source and source.suffix.lower() == ".band"):
            banddata = readers.read_openmx_bands(available.get("bandfile", source))
    elif engine == "wannier":
        if "wout" in available:
            summary["wannier"] = readers.read_wannier_report(available["wout"])
        if "bandfile" in available or (source and source.suffix.lower() != ".wout"):
            banddata = readers.read_wannier_bands(available.get("bandfile", source))
        elif source:
            summary["wannier"] = readers.read_wannier_report(source)
    else:
        raise ValueError("postprocess engine must be vasp, openmx, or wannier")
    if banddata is not None:
        if "kpoints_file" in available:
            path = readers.read_vasp_kpoints(available["kpoints_file"])
            if path["coordinate_mode"] != "fractional_reciprocal":
                raise ValueError("Band path matching requires fractional reciprocal KPOINTS")
            if banddata.kpoints is None or not np.allclose(
                path["kpoints"], banddata.kpoints, atol=1e-6, rtol=0
            ):
                raise ValueError("KPOINTS path does not match imported band rows")
            banddata = replace(
                banddata,
                labels=path["labels"],
                label_indices=path["label_indices"],
                segments=path["segments"],
            )
        if parameters.get("fermi_energy") is not None:
            banddata = replace(
                banddata,
                fermi_energy=float(parameters["fermi_energy"]),
                metadata={**banddata.metadata, "fermi_source": parameters.get("fermi_source", "explicit")},
            )
        elif banddata.fermi_energy is not None:
            banddata = replace(
                banddata,
                metadata={
                    **banddata.metadata,
                    "fermi_source": banddata.metadata.get(
                        "fermi_source", banddata.metadata.get("source_format", engine)
                    ),
                },
            )
        from ..capabilities import CapabilityError

        try:
            summary["bandgap"] = readers.band_gap(banddata)
        except CapabilityError as error:
            summary["bandgap"] = None
            summary["bandgap_unavailable_reason"] = str(error)
        result = _band_observable(banddata, summary=summary, source=engine)
        return ObservableResult(
            {**arrays, **result.arrays}, {**dims, **result.dims}, {**units, **result.units}, result.metadata
        )
    if not summary:
        raise ValueError("No supported calculation results found")
    return ObservableResult(arrays, dims, units, metadata={"summary": summary, "source": engine})
