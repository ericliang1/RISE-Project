Final results package -- super-emitter benchmark (4-8 masts, q >= 100 kg/h)
Generated 2026-07-31, branch super-emitter.

tables_se48.pdf     Table I (physics-only + 3 architectures with/without maps,
                    3 seeds, paired bootstrap CIs) and Table II (component
                    ladder), with benchmark spec and provenance notes.
fig_data_dist.*     Region-radius distributions, all 2000 test scenarios,
                    measured wind, seed 1. Solid = with images, dashed = without.
fig_data_masts.*    Median 90% region radius vs mast count (4-8), seed 1.

All numbers trace to per-scenario audit npz files in
/projectnb/rise-tower/eric1/csr-data-se48 (sizes, covered,
packed masks) and results/se48_table_deltas_ci.json. Regenerate figures with
scripts/paper_data_figs.py, tables with scripts/se_physics_table.py.
