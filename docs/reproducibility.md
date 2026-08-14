# Reproducibility status

The formal training entry now seeds Python, NumPy, PyTorch CPU, all visible PyTorch CUDA devices, the DataLoader generator, and Python/NumPy state in each worker. With the template default it enables deterministic CuDNN behavior and disables CuDNN benchmarking. A null training seed explicitly opts out of this setup.

These controls do not promise bitwise-identical results across hardware, accelerators, drivers, PyTorch/CUDA versions, or all third-party image/augmentation implementations. The source entry seeded before Dataset construction; the formal entry performs deterministic Dataset/preflight work first and reseeds immediately before model construction so the newly added validation reads do not consume model-initialization randomness. The formal DataLoader generator is separate from the global model RNG. Optional stain augmentation may introduce dependency-specific randomness not proven to be fully controlled.

Training retains two distinct expression-corruption draws: the model samples the flow prior, time and prior-side noise; the entry separately draws `target_noise_std * randn_like(ground_truth)` for all three expression MSE targets. Adding diagnostic random draws before those operations would change behavior and is prohibited.

Each schema-3 checkpoint records the resolved configuration, ordered genes, cell-type mapping, model dimensions, explicit fusion method, normalization artifact path, seed, package version, and runtime PyTorch/CUDA versions. The package, checkpoint writer, and inference runtime read `0.1.0` from one static `histcfm._version` source; packaging obtains the same attribute without a separate handwritten version. The training normalization is computed from training patches only and stored at `artifacts/histology_normalization.npy`; validation and inference reuse it through checkpoint-relative provenance.

CFM inference contains a stochastic prior and noise. Formal inference uses the runtime training-seed field when provided and otherwise the checkpoint seed; it applies the same Python, NumPy, PyTorch CPU/CUDA, DataLoader generator/worker and CuDNN controls before model construction and sampling. Each invocation performs exactly one sample and records the actual seed. It does not select across seeds or implement uncertainty.

The historical multi-checkpoint script constructs one model and DataLoader and then continues consuming RNG across checkpoint iterations. Formal inference resets for a single explicit checkpoint, so identical seed/configuration targets repeatability of the formal interface but is not claimed to be byte-identical to an epoch's position in the historical batch scan.

Bitwise reproducibility remains dependent on hardware, CUDA/cuDNN, PyTorch and image dependencies. The formal evaluator and thin CLI are implemented, and contract tests cover evaluator arithmetic/alignment, JSON finiteness, CLI arguments, strict configuration fields, and the committed synthetic demo. On 2026-08-13, an independent server environment passed all 75 tests and the complete synthetic workflow. WSL remains limited to static checks and is not used as evidence for GPU execution. No runtime estimate or synthetic metric is claimed.

The independently created `histcfm` environment used Python 3.10.14, PyTorch
2.1.1, torchvision 0.16.1, the PyTorch CUDA 12.1 runtime, and NumPy 1.26.4,
plus the directly audited data, evaluation, and test dependencies declared in
`environment.yml`. It was constructed from that declaration rather than
cloned from a historical research environment. Environment imports and CUDA
preflight passed before the workflow ran.

The formal UNI store is deterministic JSON+NPY input: a contiguous canonical
key map and finite float16/float32 matrix. Preflight rejects any missing key or
row with L2 norm at or below `1e-12`; it does not repair feature data. Runtime
uses the same loader and retains L2 normalization. No pickle index, online UNI
encoder, weight download, or feature computation participates in the public
path.

The public synthetic demo is generated with seed `20260813`. It contains 96
cells, 24 genes, four cell types, six train patches, four validation patches,
and ten 1024-dimensional float16 feature rows. Those rows are created by the
standard-library generator's pseudorandom arithmetic, not by UNI or another
model. Committed checksums make accidental data drift visible; they do not
assert model-result reproducibility.

The final `0.1.0` synthetic validation used six training patches and four
validation patches. It created and strictly reloaded one schema-3 checkpoint,
and both that checkpoint and the inference metadata recorded HistCFM `0.1.0`.
Validation inference and evaluation completed for 36 cells and 24 genes.
Predictions and targets had identical cell IDs and gene order; all 14 expected
files existed. Standard-JSON parsing found no NaN or Infinity in metadata or
metrics. These facts are software-contract evidence, not biological or paper-
benchmark results.

Real-data benchmark inputs, external-model features, checkpoints, predictions,
and metrics are not included in the public repository. The committed synthetic
results validate only software plumbing and are not biological-performance or
paper-metric claims.
