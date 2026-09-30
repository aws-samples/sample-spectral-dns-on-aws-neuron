# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Offscreen ParaView render of the frame series: one 1280 x 720 PNG per time step.

pvbatch render_frames.py <frames.pvd> <frames_stats.json> <png_dir> [--percentile 90] [--iso VALUE]
                         [--iso-frame-t 9.0] [--iso2 VALUE [--iso2-opacity 0.18]] [--width 1280 --height 720] [--subtitle TEXT]

Scene: an isosurface of vorticity_magnitude at the given percentile of the iso-frame's distribution
(default the 90th percentile of the t = 9 frame, read from frames_stats.json), coloured by |u| on a
fixed range (0 to the iso-frame's maximum speed), the box outline, a time caption, a fixed camera
looking into the periodic box. With --iso2 a second, translucent isosurface of vorticity magnitude in a
flat colour is drawn underneath, so a level the early frames reach (the initial field peaks at 2.0) keeps
the initial vortex on screen while the opaque level shows the transition. Every frame is rendered from
the same pipeline; only the time changes.
Writes render_report.json next to the PNGs (iso value, colour range, per-frame triangle counts and
seconds).
"""
import json, math, os, sys, time
from paraview.simple import *  # noqa

args = sys.argv[1:]
pvd, stats_path, outdir = args[0], args[1], args[2]
opt = {"--percentile": 90.0, "--iso": None, "--iso-frame-t": 9.0, "--iso2": None, "--iso2-opacity": 0.18, "--width": 1280, "--height": 720, "--subtitle": None}
for i in range(3, len(args), 2):
    opt[args[i]] = args[i + 1] if args[i] == "--subtitle" else float(args[i + 1])
W, H = int(opt["--width"]), int(opt["--height"])
os.makedirs(outdir, exist_ok=True)

stats = json.load(open(stats_path, encoding="utf-8"))
frames = stats["frames"]
ref = min(frames, key=lambda r: abs(r["t"] - opt["--iso-frame-t"]))
pkey = f"p{int(opt['--percentile'])}"
iso = float(opt["--iso"]) if opt["--iso"] is not None else ref["vorticity_magnitude"][pkey]
speed_max = ref["speed"]["max"]
L = 2 * math.pi
print(f"iso vorticity_magnitude = {iso:.4f} ({pkey} of frame t={ref['t']}), colour |u| in [0, {speed_max:.4f}]", flush=True)

reader = PVDReader(FileName=pvd)


def attempt(fn):
    """Run a ParaView call that exists only in some versions; True when it succeeded."""
    try:
        fn()
        return True
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return False


for prop in ("PointArrays", "PointArrayStatus"):   # the property name differs between ParaView versions
    if attempt(lambda: setattr(reader, prop, ["u", "vorticity_magnitude"])):
        break
calc = Calculator(Input=reader)
calc.ResultArrayName = "speed"
calc.Function = "mag(u)"
contour = Contour(Input=calc)
contour.ContourBy = ["POINTS", "vorticity_magnitude"]
contour.Isosurfaces = [iso]
contour.ComputeNormals = 1
contour.ComputeScalars = 1
outline = Outline(Input=reader)
iso2 = None if opt["--iso2"] is None else float(opt["--iso2"])
if iso2 is not None:
    contour2 = Contour(Input=reader)
    contour2.ContourBy = ["POINTS", "vorticity_magnitude"]
    contour2.Isosurfaces = [iso2]
    contour2.ComputeNormals = 1
    contour2.ComputeScalars = 0

view = CreateView("RenderView")
view.ViewSize = [W, H]
view.OrientationAxesVisibility = 0
attempt(lambda: setattr(view, "UseColorPaletteForBackground", 0))
view.Background = [0.04, 0.045, 0.07]

disp = Show(contour, view)
ColorBy(disp, ("POINTS", "speed"))
lut = GetColorTransferFunction("speed")
for preset in ("Inferno (matplotlib)", "Inferno", "Black-Body Radiation"):   # the name differs between 5.13 and 6.x
    if attempt(lambda: lut.ApplyPreset(preset, True)):
        break
lut.AutomaticRescaleRangeMode = "Never"
lut.RescaleTransferFunction(0.0, speed_max)
disp.Specular = 0.25
bar = GetScalarBar(lut, view)
bar.Title = "|u|"
bar.ComponentTitle = ""
bar.HorizontalTitle = 1
bar.TitleColor = [1, 1, 1]; bar.LabelColor = [1, 1, 1]
bar.TitleFontSize = 18; bar.LabelFontSize = 14
bar.WindowLocation = "Any Location"
bar.Position = [0.90, 0.25]
bar.ScalarBarLength = 0.5
disp.SetScalarBarVisibility(view, True)

if iso2 is not None:
    d2 = Show(contour2, view)
    d2.ColorArrayName = ["POINTS", ""]   # flat colour, no scalar mapping (ColorBy(d2, None) is rejected by 6.x)
    d2.AmbientColor = [0.55, 0.70, 0.95]; d2.DiffuseColor = [0.55, 0.70, 0.95]
    d2.Opacity = float(opt["--iso2-opacity"])
    d2.Specular = 0.0
odisp = Show(outline, view)
odisp.AmbientColor = [0.6, 0.65, 0.8]; odisp.DiffuseColor = [0.6, 0.65, 0.8]
odisp.LineWidth = 1.5

caption = Text(Text="t = 0.0")
cdisp = Show(caption, view)
cdisp.WindowLocation = "Upper Left Corner"
cdisp.FontSize = 30
cdisp.Color = [1, 1, 1]
sub = Text(Text=(opt["--subtitle"] or f"Taylor-Green vortex, Re 1600, {stats['N']}^3 pseudo-spectral DNS on AWS Neuron") + f".  Isosurface: vorticity magnitude {iso:.2f}, colour: speed" + ("" if iso2 is None else f"; translucent: {iso2:.2f}"))
sdisp = Show(sub, view)
sdisp.WindowLocation = "Lower Left Corner"
sdisp.FontSize = 16
sdisp.Color = [0.85, 0.85, 0.9]

c = L / 2
view.CameraFocalPoint = [c, c, c]
view.CameraPosition = [c + 2.05 * L, c - 1.75 * L, c + 1.25 * L]
view.CameraViewUp = [0, 0, 1]
view.CameraViewAngle = 30

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(reader.TimestepValues) if hasattr(reader.TimestepValues, "__len__") else [reader.TimestepValues]
report = {"iso": iso, "iso_percentile": opt["--percentile"], "iso_frame_t": ref["t"], "iso2": iso2, "speed_range": [0.0, speed_max],
          "resolution": [W, H], "frames": []}
T0 = time.perf_counter()
for i, t in enumerate(times):
    t0 = time.perf_counter()
    scene.AnimationTime = t
    view.ViewTime = t
    caption.Text = f"t = {t:.1f}"
    Render(view)
    png = os.path.join(outdir, f"frame_{i:04d}.png")
    SaveScreenshot(png, view, ImageResolution=[W, H])
    ncells = contour.GetDataInformation().GetNumberOfCells()
    row = {"index": i, "t": t, "png": os.path.basename(png), "triangles": int(ncells), "seconds": round(time.perf_counter() - t0, 2)}
    if iso2 is not None:
        row["triangles_iso2"] = int(contour2.GetDataInformation().GetNumberOfCells())
    report["frames"].append(row)
    print(json.dumps(row), flush=True)
report["total_seconds"] = round(time.perf_counter() - T0, 1)
json.dump(report, open(os.path.join(outdir, "render_report.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({"RENDER": len(times), "seconds": report["total_seconds"], "iso": iso}), flush=True)
