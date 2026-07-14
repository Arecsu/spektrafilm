#!/usr/bin/env python3
"""Export a spektrafilm release into a data pack for the native darktable module.

This is the version-upgrade mechanism: when a new spektrafilm release comes out,

    pip install <new spektrafilm>          (or pip install -e <checkout>)
    python spektrafilm_export_data.py -o ~/.config/darktable/spektrafilm

and the darktable module picks up the new profiles / data on restart. The
module itself contains only the *algorithms* (which track the spektrafilm
model version recorded in pack.json); all measured data lives in this pack.

Pack layout:
    pack.json           model constants, CMFS, spectral locus, illuminant SPDs,
                        dichroic filter curves, neutral print filter database
    spectra_lut.f32     hanatos2025 irradiance spectra LUT (header + float32)
    profiles/*.json     verbatim spektrafilm stock profiles (CC BY-SA 4.0)
"""

import argparse
import json
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np


def _widen3(v):
    """[x] -> [x, x, x]; None-padded singles ([x, None, None]) also collapse."""
    v = [x for x in v if x is not None]
    return v * 3 if len(v) == 1 else v


def _grain_export(params):
    gr = params.film_render.grain
    out = {
        "rms_granularity": _widen3(gr.rms_granularity),
        "uniformity": _widen3(gr.uniformity),
        "density_min": _widen3(gr.density_min),
    }
    if hasattr(gr, "particle_scale_sublayers"):
        out["particle_scale_sublayers"] = list(gr.particle_scale_sublayers)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="output pack directory")
    args = ap.parse_args()

    try:
        import spektrafilm  # noqa: F401
        from importlib.metadata import version as dist_version
        import importlib.resources as pkg_resources
        from spektrafilm.config import SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS, LOG_EXPOSURE
        from spektrafilm.model.illuminants import standard_illuminant
        from spektrafilm.model.color_filters import DichroicFilters
        from spektrafilm.utils.io import read_neutral_print_filters
        from spektrafilm.utils.spectral_upsampling import _load_hanatos2025_spectra_lut
        from spektrafilm.utils.gamut_compression import spectral_locus_xy
    except ImportError as err:
        print(f"error: spektrafilm must be importable ({err})", file=sys.stderr)
        print("hint:  uv run --python 3.13 --with 'spektrafilm@git+https://github.com/andreavolpato/spektrafilm@dev' python3 export_pack.py -o ~/.config/darktable/spektrafilm", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir(exist_ok=True)

    wl = SPECTRAL_SHAPE.wavelengths
    version = dist_version("spektrafilm")

    # --- illuminants (normalized SPDs on the model wavelength grid) ---------
    illuminant_names = ["D50", "D55", "D65", "D75", "T", "TH-KG3", "TH-KG3-L", "K75P"]
    illuminants = {}
    for name in illuminant_names:
        try:
            illuminants[name] = np.asarray(standard_illuminant(name), dtype=float).tolist()
        except Exception as err:  # keep the pack usable if one SPD is missing
            print(f"warning: skipping illuminant {name}: {err}", file=sys.stderr)

    # --- enlarger dichroic filters ------------------------------------------
    filter_brands = ["thorlabs", "edmund_optics", "durst_digital_light", "custom"]
    dichroics = {}
    for brand in filter_brands:
        try:
            dichroics[brand] = np.asarray(DichroicFilters(brand=brand).filters, dtype=float).tolist()
        except Exception as err:
            print(f"warning: skipping dichroic brand {brand}: {err}", file=sys.stderr)

    # --- neutral print filter database --------------------------------------
    try:
        neutral_filters = read_neutral_print_filters()
    except FileNotFoundError:
        neutral_filters = {}

    # --- per-film digested render defaults -----------------------------------
    # Stock-specific tuning (DIR coupler gamma matrices, halation presets)
    # lives in spektrafilm's params_builder. Exporting the digested values per
    # film keeps that tuning in the data pack so the native module needs no
    # hardcoded per-stock tables and upgrades cleanly with new releases.
    from spektrafilm.runtime.params_builder import init_params, digest_params

    film_render_defaults = {}
    for res in pkg_resources.files("spektrafilm.data.profiles").iterdir():
        if not res.name.endswith(".json"):
            continue
        stock = res.name[:-5]
        with pkg_resources.as_file(res) as p:
            info = json.loads(Path(p).read_text()).get("info", {})
        if info.get("stage") != "filming":
            continue
        target_print = info.get("target_print") or "kodak_portra_endura"
        try:
            params = digest_params(init_params(film_profile=stock, print_profile=target_print))
        except Exception as err:
            print(f"warning: could not digest defaults for {stock}: {err}", file=sys.stderr)
            continue
        dc = params.film_render.dir_couplers
        ha = params.film_render.halation
        if getattr(params.film, "is_bw", False):
            # single emulsion: upstream uses gamma_samelayer_rgb[0] only; on the
            # widened 3-channel profile all channels must behave identically
            g0 = dc.gamma_samelayer_rgb[0]
            dc.gamma_samelayer_rgb = (g0, g0, g0)
            dc.gamma_interlayer_r_to_gb = (0.0, 0.0)
            dc.gamma_interlayer_g_to_rb = (0.0, 0.0)
            dc.gamma_interlayer_b_to_rg = (0.0, 0.0)
        film_render_defaults[stock] = {
            "dir_couplers": {
                "gamma_samelayer_rgb": list(dc.gamma_samelayer_rgb),
                "gamma_interlayer_r_to_gb": list(dc.gamma_interlayer_r_to_gb),
                "gamma_interlayer_g_to_rb": list(dc.gamma_interlayer_g_to_rb),
                "gamma_interlayer_b_to_rg": list(dc.gamma_interlayer_b_to_rg),
                "diffusion_size_um": dc.diffusion_size_um,
                "diffusion_tail_um": dc.diffusion_tail_um,
                "diffusion_tail_weight": dc.diffusion_tail_weight,
                # Langmuir saturating couplers (spektrafilm dev/0.4+); absent
                # on 0.3.x, in which case the engine uses the linear model
                **({"langmuir_donor_k_rgb": list(dc.langmuir_donor_k_rgb),
                    "langmuir_receiver_k_rgb": list(dc.langmuir_receiver_k_rgb)}
                   if hasattr(dc, "langmuir_donor_k_rgb") else {}),
            },
            "grain": _grain_export(params),
            "halation": {
                "strength": list(ha.halation_strength),
                "first_sigma_um": list(ha.halation_first_sigma_um),
                "scatter_core_um": list(ha.scatter_core_um),
                "scatter_tail_um": list(ha.scatter_tail_um),
                "scatter_tail_weight": list(ha.scatter_tail_weight),
            },
        }

    pack = {
        "pack_format": 1,
        "spektrafilm_version": version,
        "film_render_defaults": film_render_defaults,
        "wavelengths": np.asarray(wl, dtype=float).tolist(),
        "log_exposure": np.asarray(LOG_EXPOSURE, dtype=float).tolist(),
        "cmfs": np.asarray(STANDARD_OBSERVER_CMFS[:], dtype=float).tolist(),
        "spectral_locus_xy": np.asarray(spectral_locus_xy(), dtype=float).tolist(),
        "illuminants": illuminants,
        "dichroic_filters": dichroics,
        "neutral_print_filters": neutral_filters,
    }
    def _sanitize(obj):
        """JSON has no NaN/Inf — emit null instead (readers treat null as NaN)."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    (out / "pack.json").write_text(json.dumps(_sanitize(pack)))
    print(f"wrote {out / 'pack.json'} (spektrafilm {version})")

    # --- hanatos2025 spectra LUT ---------------------------------------------
    lut = np.ascontiguousarray(_load_hanatos2025_spectra_lut(), dtype=np.float32)
    with open(out / "spectra_lut.f32", "wb") as fh:
        fh.write(b"SFSL")
        fh.write(struct.pack("<iii", *lut.shape))
        fh.write(lut.tobytes())
    print(f"wrote {out / 'spectra_lut.f32'} shape {lut.shape}")

    # --- stock profiles ------------------------------------------------------
    # colour profiles are copied verbatim; single-emulsion B&W profiles
    # (channel_model == "bw", spektrafilm dev/0.4+) are widened to the 3-channel
    # layout the C engine expects: per-channel arrays are triplicated and
    # channel_density is divided by 3 so the spectral sum over channels equals
    # the single emulsion's density spectrum. info.channel_model stays "bw" so
    # the module can couple the grain across channels.
    profile_dir = pkg_resources.files("spektrafilm.data.profiles")
    n = nbw = 0
    for res in profile_dir.iterdir():
        if not res.name.endswith(".json"):
            continue
        with pkg_resources.as_file(res) as p:
            prof = json.loads(p.read_text())
        if prof.get("info", {}).get("channel_model") == "bw":
            d = prof["data"]
            # BW development-time families (push/pull data): collapse to the
            # representative middle development time, exactly like upstream's
            # select_development_time(None) does, BEFORE channel widening
            times = d.get("development_time") or []
            curves = d.get("density_curves") or []
            if len(times) > 1 and curves and isinstance(curves[0], list) \
               and len(curves[0]) == len(times):
                idx = (len(times) - 1) // 2
                d["density_curves"] = [[row[idx]] for row in curves]
                base = d.get("base_density")
                if base and isinstance(base[0], list):
                    d["base_density"] = [row[idx] for row in base]
                layers = d.get("density_curves_layers")
                if layers and isinstance(layers[0], list) and isinstance(layers[0][0], list):
                    d["density_curves_layers"] = [[[lay[idx]] for lay in row] for row in layers]
                d["development_time"] = [times[idx]]
            for key in ("log_sensitivity", "density_curves"):
                if key in d and d[key] and isinstance(d[key][0], list) and len(d[key][0]) == 1:
                    d[key] = [row * 3 for row in d[key]]
            if "channel_density" in d and d["channel_density"] \
               and isinstance(d["channel_density"][0], list) and len(d["channel_density"][0]) == 1:
                d["channel_density"] = [
                    [None if row[0] is None else row[0] / 3.0] * 3
                    for row in d["channel_density"]
                ]
            nbw += 1
        (out / "profiles" / res.name).write_text(json.dumps(prof))
        n += 1
    print(f"copied {n} profiles ({nbw} B&W widened to 3 channels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
