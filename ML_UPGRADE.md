# ML upgrade: what changed, what to do next

## Added in this pass
| Item | Where | Status |
|---|---|---|
| Multi-class U-Net (bg/wall/door/window), augmentation, Dice+CE loss, ONNX export | `training/` | Written, syntax-checked only (no torch in the build sandbox) |
| Door/window detection from the same network (tight boxes along the wall) | `perception/segsymbols.py`, `PerceptionConfig.wall_model_door_channel / window_channel` | Unit-tested with a fake model |
| Accuracy harness (wall F1, opening P/R, scale error) | `evaluation/metrics.py` | Sanity-tested |
| 3D: wall ends extended only where they meet another wall (free ends no longer overshoot) | `export3d/builder.py` | Unit-tested |
| 3D: sub-mm height stagger against z-fighting at coplanar corner tops | `export3d/builder.py` | Logic only; confirm visually |

## Usage
```
pip install -e ".[train]"
python -m training.train --data data/train --val data/val --out walls.onnx
python examples/run_pipeline.py plan.png --wall-model walls.onnx   # set door/window channels 2/3 in PerceptionConfig
```
```python
PerceptionConfig(wall_model_path="walls.onnx", wall_model_channel=1,
                 wall_model_door_channel=2, wall_model_window_channel=3)
```

## Still open (needs data or a browser, not more heuristics)
1. No trained weights exist yet. Accuracy gain is zero until you train on CubiCasa5k (+ your own plans) and validate.
2. Rasterise CubiCasa5k SVG labels into `images/` + `labels/` PNG pairs (not included).
3. Manhattan snapping still distorts diagonal/curved walls; add 45/135 to `wall_angles_deg` and `snap_angles_deg` and test.
4. Wall thickness clamp (0.10-0.35 m), fixed heights, single page: unchanged.
5. Viewer was never rendered in a browser; test it, then add door leaves, window frames, room-type labels.
6. Shapely/ezdxf/PyMuPDF were unavailable here, so the topology/scale stages and the full test suite were NOT run.
