# Uncertainty-Aware Physics Guidance for Methane Source Localization

Code for the paper *"Uncertainty-Aware Physics Guidance for Methane
Source Localization"*.

## How it works

- **CH4-T simulator** (`simulator/`): Gaussian-plume methane leak
  scenarios on a 500 m × 500 m site — one super-emitter leak
  (100–500 kg/h), 4–8 point-sensor masts sampling methane and wind every
  minute for 30 minutes, drifting wind, and realistic methane-sensor and
  anemometer error.
- **Localization** (`localization/`): three permutation-invariant
  networks (DeepSets, GNN, Set Transformer) predict a probability map of
  the leak location over a 64 × 64 grid; split conformal calibration
  turns the map into a region guaranteed to contain the true source 90%
  of the time.
- **Physics guidance**: from the readings and the measured wind, three
  physics-based input maps are computed — how well a leak at each cell
  explains the readings (source evidence), how visible each cell is to
  the masts (sensor visibility), and how stable the evidence is across
  8 Monte-Carlo draws of plausible true winds (wind sensitivity). The
  maps feed a small zero-initialized convolutional head (10,433
  parameters) added to each network.
- **Result**: coverage holds near 90% while median region radii fall
  from 97–100 / 68–69 / 72–75 m to 73–75 / 61–62 / 64–65 m, and the
  share of scenarios meeting the 50 m localization target rises from
  0–12% to 16–30%.

## Run

```bash
bash simulator/generate_data.sh       # data + gate + physics maps (GPU)
bash localization/train_chain.sh 0    # training, chain 0 (GPU)
bash localization/train_chain.sh 1    # training, chain 1 (GPU)
python localization/table1_from_audits.py
```

Table 1 straight from the committed results (no data or GPU needed):

```bash
python localization/table1_from_results_json.py
```

## Layout

```
simulator/       plume model, scenario generation, exact posterior, datagen
localization/    networks, physics maps, conformal wrapper, training,
                 table scripts
config/          the one config: all settings and seeds
results/         the Table 1 numbers (paired_wind.json)
```

Data generation is deterministic from the seeds in
`config/default.yaml`, so the benchmark regenerates identically.
`paths.data_dir` (default `data/`, override with `CSR_DATA_DIR`) sets
where the generated files live. Requires Python 3.12, PyTorch (CUDA),
numpy, pyyaml.
