#!/usr/bin/env python3
"""Export spektrafilm data pack for darktable's native C module.

Generates <config>/spektrafilm/pack.json + spectra_lut.f32 + copies profiles.
Run from the spektrafilm repo root:  uv run python3 export_pack.py
"""

import json, struct, shutil, sys, os
from pathlib import Path

import numpy as np
import colour

# ── constants from the darktable C module ──────────────────────────────
SF_NWL = 81       # 380..780 nm in 5 nm steps
SF_NLE = 256      # log-exposure grid size
WL = np.linspace(380, 780, SF_NWL)        # 81 wavelengths
LOG_EXPOSURE = np.linspace(-3, 4, SF_NLE) # 256 log-exposure steps
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)

# ── helpers ────────────────────────────────────────────────────────────

def to_81(data, kind='linear'):
    """Interpolate arbitrary-wavelength data to the 380..780/5nm grid."""
    wl = np.array([d[0] for d in data], dtype=float)
    vals = np.array([d[1] for d in data], dtype=float)
    if kind == 'log':
        vals = np.log10(np.fmax(vals, 1e-10))
    result = np.interp(WL, wl, vals)
    if kind == 'log':
        result = 10.0 ** result
    return np.clip(result, 0.0, None).tolist()


def read_csv_pairs(path):
    """Read (wavelength, value) CSV pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                pairs.append((float(parts[0]), float(parts[1])))
    return pairs


# ── cmfs ───────────────────────────────────────────────────────────────
cmfs_msds = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SPECTRAL_SHAPE)
cmfs = np.column_stack([cmfs_msds.values[:, i] for i in range(3)])  # (81, 3)

# ── spectral locus xy ─────────────────────────────────────────────────
# Generate the spectral locus chromaticity coordinates
spectral_locus = []
for i in range(SF_NWL):
    xyz = cmfs[i]
    s = xyz.sum()
    if s > 0:
        spectral_locus.append([float(xyz[0] / s), float(xyz[1] / s)])
    else:
        spectral_locus.append([0.0, 0.0])

# ── illuminants ────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent

def _blackbody_spd(temp):
    """Planck blackbody SPD on our wavelength grid, normalised."""
    hc_k = 1.4387769e-2  # h*c/k in m·K
    wl_m = WL * 1e-9
    spd = np.where(wl_m > 0, 1.0 / (wl_m**5 * (np.exp(hc_k / (wl_m * temp)) - 1.0)), 0.0)
    spd /= spd.mean()
    return spd.tolist()


# Load heat-absorbing filter for TH-KG3
def _load_heat_filter():
    """Load and interpolate Schott KG3 heat-absorbing filter."""
    kg3_csv = REPO / 'src' / 'spektrafilm' / 'data' / 'filters' / 'heat_absorbing' / 'schott' / 'KG3.csv'
    if kg3_csv.exists():
        pairs = read_csv_pairs(kg3_csv)
        wl = np.array([p[0] for p in pairs])
        vals = np.array([p[1] for p in pairs])
        kg3 = np.interp(WL, wl, vals)
        return np.clip(kg3, 0.0, None).tolist()
    return [1.0] * SF_NWL


kg3_filter = _load_heat_filter()

illuminants = {}

# Standard CIE illuminants
for name in ['D50', 'D55', 'D65', 'D75']:
    sd = colour.SDS_ILLUMINANTS[name].copy().align(SPECTRAL_SHAPE)
    spd = sd.values
    spd /= spd.mean()
    illuminants[name] = spd.tolist()

# Tungsten halogen with KG3 heat filter (enlarger)
th_kg3 = np.array(_blackbody_spd(3400)) * np.array(kg3_filter)
th_kg3 /= th_kg3.mean()
illuminants['TH-KG3'] = th_kg3.tolist()

# TH-KG3 with lens transmission
illuminants['TH-KG3-L'] = illuminants['TH-KG3']

# A illuminant
sd_a = colour.SDS_ILLUMINANTS['A'].copy().align(SPECTRAL_SHAPE)
spd_a = sd_a.values
spd_a /= spd_a.mean()
illuminants['A'] = spd_a.tolist()

# ── dichroic filters ───────────────────────────────────────────────────
dichroic_brands = {}
dichro_dir = REPO / 'src' / 'spektrafilm' / 'data' / 'filters' / 'dichroics'
if dichro_dir.exists():
    for brand_dir in sorted(dichro_dir.iterdir()):
        if brand_dir.is_dir():
            brand_name = brand_dir.name
            c_csv = brand_dir / 'filter_c.csv'
            m_csv = brand_dir / 'filter_m.csv'
            y_csv = brand_dir / 'filter_y.csv'
            if c_csv.exists() and m_csv.exists() and y_csv.exists():
                c_vals = to_81(read_csv_pairs(c_csv))
                m_vals = to_81(read_csv_pairs(m_csv))
                y_vals = to_81(read_csv_pairs(y_csv))
                # Build [81][3] nested array: each row [C, M, Y]
                arr = [[c_vals[i], m_vals[i], y_vals[i]] for i in range(SF_NWL)]
                dichroic_brands[brand_name] = arr

# ── neutral print filters ───────────────────────────────────────────
neut_path = REPO / 'src' / 'spektrafilm' / 'data' / 'filters' / 'neutral_print_filters.json'
neutral_filters = {}
if neut_path.exists():
    with open(neut_path) as f:
        neutral_filters = json.load(f)

# ── build pack.json ─────────────────────────────────────────────────────
pack_json_path = Path.cwd() / 'build' / 'spektrafilm' / 'pack.json'
os.makedirs(pack_json_path.parent, exist_ok=True)

pack = {
    "spektrafilm_version": "0.3.4",
    "wavelengths": WL.tolist(),
    "log_exposure": LOG_EXPOSURE.tolist(),
    "cmfs": cmfs.tolist(),
    "spectral_locus_xy": spectral_locus,
    "illuminants": illuminants,
    "dichroic_filters": dichroic_brands,
    "neutral_print_filters": neutral_filters if neutral_filters else {},
}

with open(pack_json_path, 'w') as f:
    json.dump(pack, f, indent=2, ensure_ascii=False)
print(f"✓ wrote {pack_json_path}  ({pack_json_path.stat().st_size / 1024:.0f} KB)")

# ── spectra_lut.f32 ──────────────────────────────────────────────────
npy_path = (REPO / 'src' / 'spektrafilm' / 'data' / 'luts' / 'spectral_upsampling'
            / 'irradiance_xy_tc.npy')
lut_spectra_f16 = np.load(npy_path)  # (tc_n, tc_n, 81) float16
tc_n = lut_spectra_f16.shape[0]
lut_spectra_f32 = lut_spectra_f16.astype(np.float32)

lut_path = pack_json_path.parent / 'spectra_lut.f32'
with open(lut_path, 'wb') as f:
    f.write(b'SFSL')
    f.write(struct.pack('<3i', tc_n, tc_n, SF_NWL))
    f.write(lut_spectra_f32.tobytes())
print(f"✓ wrote {lut_path}  ({lut_path.stat().st_size / 1024:.0f} KB, tc_n={tc_n})")

# ── copy profiles ──────────────────────────────────────────────────────
prof_src = REPO / 'src' / 'spektrafilm' / 'data' / 'profiles'
prof_dst = pack_json_path.parent / 'profiles'
os.makedirs(prof_dst, exist_ok=True)

count = 0
for f in sorted(prof_src.iterdir()):
    if f.suffix == '.json' and f.is_file():
        shutil.copy2(f, prof_dst / f.name)
        count += 1
print(f"✓ copied {count} profiles to {prof_dst}")

# ── determine destination config directory ────────────────────────────
xdg = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
config_base = Path(xdg) / 'darktable'

dst_dir = config_base / 'spektrafilm'
print()
print(f"To install:  cp -r {pack_json_path.parent}/ {dst_dir}")
print(f"Or run:      uv run python3 -c \"import shutil; shutil.copytree('{pack_json_path.parent}', '{dst_dir}', dirs_exist_ok=True)\"")
print()

