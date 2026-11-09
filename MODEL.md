# Model card: D-FINE S, aphid infestation regions

**Model.** D-FINE S (HGNetV2 backbone), fine-tuned from the public COCO checkpoint of
[D-FINE](https://github.com/Peterande/D-FINE) (Apache-2.0). One class: `aphid_infestation_region`.

**Files.** The release carries `dfine-s-aphid-v2.onnx` (41 MB), its companion
`dfine-s-aphid-v2.onnx.provenance.json` and `designation-manifest.json`. Put the three files side by side in
the inference service's model directory (see the README). The provenance file records the SHA-256 of the
network, the input size and the tiling grid; the service refuses to load a network whose bytes do not match.

**Input and output.** Input `images`: float32, N×3×1280×1280, RGB, values 0–1 (no mean/std normalisation);
`orig_target_sizes`: int64, N×2, `[[1280, 1280]]`. Output per tile: `labels` (N×300), `boxes` (N×300×4, x0 y0
x1 y1 in tile pixels), `scores` (N×300). The service cuts each picture into 1280×1280 tiles with a stride of
1024 (12 tiles for a 4056×3040 picture), keeps the boxes above the region threshold (0.40) and merges them
across tiles with a fixed union rule. The exported graph is opset 16, fp32, with a dynamic batch axis.

**Runtimes.** ONNX Runtime on CPU or CUDA as it is. On a Jetson the service runs a TensorRT engine built from
this file (TensorRT 10.3 on JetPack 6.2); the engine file then needs its own provenance record and an entry in
the designation manifest, as the README describes.

**Training and evaluation.** Trained on 32 frames with 148 labelled regions and evaluated on 13 frames with 54
regions from the same capture session; there is no separate test set. At this data size the network is not a
competitive, deployable detector; it is a proof of the pipeline. The S network published here finds 15 of the
54 validation regions; the L network trained on the same data finds 23 of the 54, at about 2.5 false regions
per frame.

**Intended use.** Detection of aphid infestation regions on high-resolution pictures of brassica plants taken
from above, as the detector of the pest-session pipeline in this repository. Pictures of other crops, other
pests or other cameras need re-training.

**Licence.** Apache-2.0, like the code. The network is a derivative of D-FINE's COCO checkpoint; see NOTICE.
