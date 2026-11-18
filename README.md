# rqf-ssl

Self-supervised future prediction with Recursive Quadrature Filters (RQFs).

This repository contains the shared-channel RQF encoder, learned window pooler,
future-window InfoNCE objective, dense future log-power objective,
coordinate-aware spatial readout, Euclidean alignment, and training examples.
The end-to-end examples are binary abnormality detection on TUAB and ten-class
Speech Commands (SC10), each with the windowed and the dense objective, and
four-class motor imagery on BCI-IV-2a with in-domain or cross-corpus
pretraining.
Dataset files and pretrained weights are not distributed here.

## Installation

Use Python 3.12 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[preprocessing,test]'
```

For GPU training, install a CUDA-enabled PyTorch build appropriate for your system,
then install the optional Triton dependency on Linux:

```bash
python -m pip install -e '.[cuda]'
```

Triton also needs a C compiler and development headers at runtime. On Ubuntu
or Debian, install them with `sudo apt-get install gcc libc6-dev`.

The implementation includes full-BPTT custom scan kernels in
`src/rqf_ssl/_core/ops/`, including prospective-input and planar BF16 paths.
Eligible CUDA inputs select these paths automatically. CPU execution uses a
PyTorch fallback; use it for small checks, not full-length training.
Kernel chunking preserves gradients across chunk boundaries; it does not truncate
backpropagation through time. GPU numerical tests are included but require a CUDA
device and Triton. Both TUAB and SC10 have been GPU-verified. The SC10
end-to-end check used Python 3.12, PyTorch 2.8.0 with CUDA 12.8, and an NVIDIA
L40S, including 50 SSL epochs and 200 downstream epochs for each of the four
pretrained/paired-initialization and frozen/full-fine-tuning configurations.

Set `RQF_SSL_DISABLE_TRITON=1` before starting Python to force the PyTorch scan
path. This is useful for diagnostics and can require more memory and time.

## Examples

### Prepare recordings

Obtain TUAB through its data provider and retain its official evaluation partition.
Prepare a CSV next to your EDF files with these exact columns, in this order:

```csv
path,patient_id,split,label
recordings/train_normal.edf,train_patient_1,train,0
recordings/val_abnormal.edf,val_patient_1,val,1
recordings/test_normal.edf,test_patient_1,test,0
```

These filenames are illustrative, not a usable dataset. Labels are `0` (normal)
and `1` (abnormal). Paths must be relative to the CSV directory and stay inside
it. Splits are `train`, `val`, and `test`; a patient may occur in only one split.
Validation and test need both classes for AUROC. Use all recordings belonging to
a patient in that patient's assigned split. For a new training/validation split,
`rqf_ssl.data.patient_train_val_split` provides a seeded 80/20 patient partition;
do not apply it to the official test patients.

```bash
mkdir -p data runs
python examples/preprocess.py /path/to/recordings.csv data/tuab
```

Preprocessing resamples referential EEG to 128 Hz, derives 16 ordered TCP bipolar
channels, extracts complete non-overlapping 100-second segments, and divides each
channel by its segment's 95th absolute-value percentile plus `1e-8`.
No notch filter is applied. Common `EEG ...-REF` / `-LE` names and T7/T8/P7/P8
aliases are accepted. Missing required channels fail explicitly. Incomplete
trailing segments are omitted. The output is a manifest and float32 `.npy` arrays
of shape `[12800, 16]`. The destination must not already exist.

The supplied preprocessing and patient-split helper do not establish the exact
recording membership of the paper's prepared data. Reproducing a reported score
requires matching that membership as well as the training configuration.

### Prepare Speech Commands SC10

Download the official [Speech Commands v0.02 archive](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz), whose dataset documentation is [here](https://www.tensorflow.org/datasets/catalog/speech_commands). Prepare the archive directly; do not substitute a transformed twelve-class TFDS variant.

```bash
mkdir -p data runs
python examples/preprocess_sc10.py speech_commands_v0.02.tar.gz data/sc10
```

This recipe uses the ordered ten words `yes,no,up,down,left,right,on,off,stop,go`, mono 16 kHz clips of exactly 16,000 samples, and a deterministic clip-level split. It is not Google's official speaker-disjoint split. Training normalization is fitted on the resulting train clips only. No audio or pretrained weights are bundled with this repository.

Preparation holds the complete train split in memory to calculate its single
float32 normalization mean and sample standard deviation, so use a machine with
substantial available RAM for the full archive.

### SC10 self-supervised pretraining and transfer

```bash
python examples/train_sc10.py pretrain data/sc10/manifest.json runs/sc10-ssl \
  --config examples/sc10.json --device cuda

python examples/train_sc10.py finetune data/sc10/manifest.json \
  runs/sc10-ssl/pretrained.pt runs/sc10-frozen --mode frozen --seed 0 --device cuda

python examples/train_sc10.py finetune data/sc10/manifest.json \
  runs/sc10-ssl/pretrained.pt runs/sc10-full-ft --mode full-ft --seed 0 --device cuda

python examples/train_sc10.py evaluate data/sc10/manifest.json \
  runs/sc10-frozen/best.pt --device cuda
```

The SC10 SSL representation is the deterministic mean of real/imaginary encoder states within 400-sample windows; unlike TUAB, it has no learned pooler. Downstream training transfers encoder weights only and always creates a fresh pointwise readout. For the paired initialization control, replace `pretrained.pt` with `init.pt`, keep the same downstream seed, and use a new output directory. Frozen mode does not optimize encoder parameters; full fine-tuning does.

SC10 SSL defaults are 50 epochs, batch size 32, AdamW `lr=0.001`, weight decay `0.01`, four-window future prediction, 256 negatives, temperature `0.1`, and variance weight `0.04`. Downstream defaults are 200 epochs, batch size 128, AdamW `lr=0.01`, `eps=1e-4`, no weight decay, and cosine decay (`T_max=200`, `eta_min=1e-6`); strict validation accuracy selects `best.pt`. Test clips are read only by `evaluate`.

The SSL encoder keeps its filter time constants fixed. Its validation split is not used for pretraining or checkpoint selection. Frozen downstream fitting does not optimize encoder parameters; its post-step constraint projection can still bring deliberately out-of-bound transferred filter values back into range on the first step. Full fine-tuning enables all encoder parameters, including those time constants.

The full-data SC10 check used one seed. It verifies the complete pipeline;
it does not reproduce the paper's multi-seed estimates.

### Self-supervised pretraining

```bash
python examples/train.py pretrain data/tuab/manifest.csv runs/ssl \
  --config examples/tuab.json --include-validation --device cuda
```

`--include-validation` includes validation signals without using their labels for
the SSL loss, as in the paper's TUAB setup. Without this flag, only training
signals are used. Test signals are never read during pretraining or downstream
fitting.

The default example uses four width-64 RQF layers, one-second windows, and a
one-window prediction horizon. It trains for 30 epochs with batch size 8,
AdamW (`lr=1e-4`, `weight_decay=0.05`), and gradient clipping at norm 1.
Each prediction uses 256 negatives, temperature 0.1, and a variance-hinge weight
of 0.04. Targets are detached. Weight decay applies to all optimized parameters,
including the learnable filter parameters.

At the default batch size, the sampled negative features alone occupy about
792 MiB, in addition to encoder activations and gradients. Total GPU memory use
has not yet been profiled for this export. Reducing `--batch-size` also changes
the pool from which negatives are sampled, so record any such change.

The run saves `init.pt` before any SSL updates, `last.pt` after each epoch, and
`pretrained.pt` after the final epoch. Checkpoints include model configuration and
training settings. The output directory must be new. `last.pt` is replaced as
training progresses; this release does not provide a resume command.

### Downstream training

```bash
python examples/train.py finetune data/tuab/manifest.csv \
  runs/ssl/pretrained.pt runs/frozen --mode frozen --seed 0 --device cuda

python examples/train.py finetune data/tuab/manifest.csv \
  runs/ssl/pretrained.pt runs/full-ft --mode full-ft --seed 0 --device cuda
```

Both modes transfer the encoder and learned window pooler, discard the SSL
predictor, and initialize a new spatial readout and classifier. In frozen mode,
the encoder and pooler remain fixed while the spatial readout and classifier are
trained. Full fine-tuning updates the entire downstream model.

To train the paired-initialization control, use `runs/ssl/init.pt` instead of
`runs/ssl/pretrained.pt`, with the same downstream seed and a new output directory.
This reuses the byte-identical pre-SSL representation weights. It does not draw
an independent encoder initialization.

Defaults are 20 epochs, batch size 4, AdamW with weight decay 0.01, learning rate
`2e-4` for the transferred representation and `2e-3` for the new readout,
five-epoch linear warmup followed by exponential decay of 0.99 per epoch,
and gradient clipping at norm 1. Validation AUROC selects `best.pt`;
`history.json` records the learning curves. Repeat downstream training with seeds
0, 1, and 2 to summarize variability conditional on a fixed SSL checkpoint.

### Label budgets

```bash
python examples/train.py finetune data/tuab/manifest.csv \
  runs/ssl/pretrained.pt runs/full-ft-1pct --mode full-ft --seed 0 \
  --label-fraction 0.01 --label-seed 0 --device cuda
```

`--label-fraction` trains on every segment of a subset of training patients.
Each class keeps `max(1, floor(fraction x patients))` of its patients, drawn
from one seeded generator, normal patients first. All budgets read the same
permutations, so the patients of a smaller budget are always retained by a
larger one. Validation and test patients are unchanged. The paper's sweep uses
budgets 0.01, 0.05, 0.10, and 0.25 with label seed 0 and downstream seeds 0, 1,
and 2, pairing each pretrained checkpoint with its `init.pt` under both modes.
The selected fraction, seed, and resulting patient and segment counts are
stored in each downstream checkpoint. `rqf_ssl.data.label_budget_indices`
exposes the same selection.

### Evaluate the selected checkpoint

```bash
python examples/train.py evaluate data/tuab/manifest.csv \
  runs/frozen/best.pt --device cuda --output-json runs/frozen/test.json
```

This separate command reports segment-level test AUROC and balanced accuracy.
Run it after model selection; do not use repeated test evaluations to choose a
configuration. Existing result files are not overwritten.

### Dense future prediction

Both training scripts accept `--objective dense`, which replaces windowed
prediction with dense future log-power prediction. At 256 fixed anchor
timesteps (drawn once with seed 1701), a three-layer predictor maps the real
and imaginary encoder state to the log mean pre-activation power of the final
RQF layer over a future target window. InfoNCE with 256 sampled negatives and
temperature 0.1 ranks the true future first; masked negatives never enter the
softmax. Targets are standardized per filter with statistics fitted on a fixed
subset of 256 pretraining sequences (seed 1702), before the first epoch and
again after every epoch. Refitting runs in evaluation mode, in full precision,
and does not consume the training random stream. The variance weight does not
apply to the dense objective.

```bash
python examples/train_sc10.py pretrain data/sc10/manifest.json runs/sc10-dense \
  --config examples/sc10.json --objective dense --device cuda

python examples/train.py pretrain data/tuab/manifest.csv runs/tuab-dense \
  --config examples/tuab.json --objective dense --device cuda
```

For SC10 the horizon is 1,600 samples (100 ms) and the target window 400
samples (25 ms). Negatives are drawn from every anchor of every clip in the
batch, except anchors of the same clip within 400 samples of the positive. The
other SSL defaults are those of the windowed SC10 recipe, and downstream
training is unchanged: the SC10 `finetune` and `evaluate` commands above accept
the dense checkpoint as they are.

For TUAB the horizon is 256 samples (2 s) and the target window 128 samples
(1 s). Each channel is encoded alone, and a negative is always the same channel
of a different recording in the batch. Calibration counts channel sequences, so
its 256 sequences cover 16 recordings. The other SSL defaults are those of the
windowed TUAB recipe. The dense checkpoint records `temporal_support: dense`,
which changes the downstream model: there is no window pooler, and the spatial
readout consumes the 128 real and imaginary state values of every channel at
every timestep, accumulating its temporal mean and standard deviation over
chunks of 128 steps. Frozen transfer therefore fixes the encoder alone. The
dense downstream system has 127,730 real scalars (33,664 in the encoder and
94,066 in the readout). The paper's dense TUAB comparison pretrains on training
signals only, so it omits `--include-validation`, and it is frozen-only at one
seed. Use the TUAB `finetune` and `evaluate` commands above unchanged.

The dense paths are covered by the CPU tests below. They have not yet been
run end to end on a GPU for this export.

### BCI-IV-2a motor imagery

Download `BCICIV_2a_gdf.zip` and the evaluation labels `true_labels.zip` from
the [BCI Competition IV page](https://www.bbci.de/competition/iv/), accept its
terms, and extract both. Then prepare the trials:

```bash
python examples/preprocess_bcic.py /path/to/gdf /path/to/true_labels data/bcic
```

All 5,184 trials are kept, with subjects 1-5, 6-7, and 8-9 as the training,
validation, and test splits (2,880, 1,152, and 1,152 trials). Each trial uses
the 22 EEG channels without EOG, common-average referencing, a causal
fifth-order Butterworth bandpass at 0.3-40 Hz over the six seconds from trial
onset, the four seconds from the cue, and Fourier resampling from 250 to
200 Hz, in microvolts. The output is `signals.npy` (`[5184, 22, 800]`,
channel-first), `trials.csv`, and `manifest.json`. The models read these
signals divided by 100, as in the paper's runs.

```bash
python examples/train_bcic.py pretrain data/bcic runs/bcic-ssl --device cuda

python examples/train_bcic.py finetune data/bcic runs/bcic-ssl/pretrained.pt \
  runs/bcic-ea-s0 --alignment euclidean --seed 0 --device cuda

python examples/train_bcic.py evaluate data/bcic runs/bcic-ea-s0/best.pt \
  --device cuda --output-json runs/bcic-ea-s0/test.json
```

In-domain pretraining reads only training-subject trials, without alignment,
and holds out the sixth run of each subject's second session, which the paper
used to monitor pretraining. Each trial gives eight non-overlapping
half-second window summaries, and the objective predicts the summary one
second ahead. Its 256 negatives are distinct summaries of the same channel in
other trials. Defaults are 60 epochs, batch size 240, AdamW (`lr=1e-4`,
weight decay 0.05), gradient clipping at norm 1, and a five-epoch linear
warmup followed by cosine decay towards 5% of the peak rate.

Downstream training copies the encoder alone, verbatim, from any rqf-ssl
checkpoint: in-domain, TUAB, or TUEG. A fresh pooler and spatial readout
summarize two non-overlapping two-second windows per trial. The complete model
has 112,820 real scalars (encoder 33,664, pooler 16,576, readout 62,580).
`--alignment euclidean` multiplies every trial by the inverse square root of
its session's mean spatial covariance, scaled to keep the session's mean
channel power. The matrix is estimated separately for every session,
including validation and test sessions, from its unlabeled trials; `evaluate`
applies the alignment recorded in the checkpoint. Full fine-tuning (the
default) uses AdamW with weight decay 0.01, learning rate `2e-4` for the
encoder, including its filter time constants, and `2e-3` for everything else,
with the TUAB warmup and decay schedule, batch size 32, bf16 autocast on CUDA,
and no gradient clipping. Training runs for up to 50 epochs and stops after
ten epochs without a strict improvement in validation Cohen's kappa; the first
best epoch is kept. `evaluate` reports balanced accuracy, Cohen's kappa, and
weighted F1 on the test subjects.

For the paired-initialization control, create one initialization checkpoint
and use it twice: pass it to `pretrain --init-from`, which copies its encoder
before any update, and fine-tune it directly as the untrained source. A
cross-corpus source is fine-tuned the same way, from its `pretrained.pt`. A
128 Hz TUAB or TUEG encoder keeps its tensors, including its frequency-range
buffers, inside the 200 Hz model, as in the paper.

### TUEG pretraining configuration

The paper's cross-corpus source is windowed prediction on a 3,768-hour subset
of TUEG that excludes TUAB patients. This release provides its configuration
but not its data preparation:

```bash
python examples/train.py pretrain data/tueg/manifest.csv runs/tueg \
  --config examples/tueg.json --negative-policy same_channel \
  --batch-size 64 --epochs 30 --warmup-epochs 2 --final-lr-fraction 0.05 \
  --device cuda
```

The manifest uses the TUAB format with every segment in the `train` split.
Each paper example is 100 seconds of the 16 TCP channels at 128 Hz, assembled
from 50 consecutive two-second segments, each divided by its own 95th
absolute-value percentile. `examples/tuab.json` differs only in its horizon:
the TUEG configuration predicts two windows (two seconds) ahead, with distinct
same-channel negatives from other examples and a two-epoch warmup before
cosine decay.

The BCI-IV-2a and TUEG paths are covered by the CPU tests below. They have not
yet been run end to end on a GPU for this export.

## Checks

```bash
python -m pytest -q
```

Tests create synthetic signals and a synthetic EDF, exercise preprocessing,
pretraining, frozen/full-FT transfer and test evaluation, and verify that fitting
does not open test arrays. They check the full-sequence gradients and the real-scalar
parameter counts: 33,664 for the temporal encoder and 110,002 for the complete
default downstream system. Dense tests check the anchors, the target window,
both negative samplers and their masks, the sampled-logit gradients, target
calibration, the chunked readout against whole-sequence pooling, and dense
pretraining and transfer for both datasets. BCI-IV-2a tests check the
alignment, the distinct same-channel negatives on both sampling paths, the
learning-rate schedule, the pretraining holdout, and in-domain and
cross-corpus transfer through test evaluation. CUDA tests compare the custom prospective scan with
a PyTorch recurrence and are skipped when no CUDA device is available.

The tests establish that these examples run; they do not establish reproduction
of the paper's numerical results. No access to external experiment infrastructure
is required.
