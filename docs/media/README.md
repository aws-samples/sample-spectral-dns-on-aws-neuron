# Media

`tgv512_re1600_neuron.gif` and `.mp4`: the Re = 1600 Taylor-Green vortex at 512^3, run to t = 10 on eight Inferentia2
NeuronCores (one inf2.24xlarge, 27 minutes of solver time) with the spectral state dumped every 81 steps (21 frames,
t = 0 to 9.94 every 0.497). Each frame is the solver's own checkpoint inverse-transformed in NumPy
(`tools/vtk_frames.py`) and rendered offscreen by ParaView 5.13.3 (`tools/render_frames.py` under `pvbatch`):
an opaque isosurface of vorticity magnitude at 6.45 (the 90th percentile of the t = 8.9 frame) coloured by speed on a
fixed 0 to 1.395 scale, a translucent isosurface at 1.5 that keeps the initial vortex visible before the opaque level is
reached at t = 3.5, the box outline and a time caption; 1280 x 720, 6 frames per second, the last frame held for
1.5 s in the GIF. The opaque surface grows from 0 to 18.2 million triangles over the run (`render_report_512_8.json`).
The field-space kinetic energy and enstrophy of the frames agree with the solver's spectral statistics to 1.8e-7 and
1.6e-7 (Parseval check of `tools/vtk_frames.py`).

Reproduce on an inf2.24xlarge with the headless ParaView build unpacked under `$PV`:

```bash
source scripts/env.sh
scripts/render_hero.sh 512 8 81 "$PV" 10 1.5          # solve with checkpoints, frames, PNGs (about 70 minutes)
tools/make_video.sh build/n512_r8/png out 6 tgv512_re1600_neuron
```
