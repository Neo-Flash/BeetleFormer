# BeetleFormer reproducibility record

This directory contains the exact sequence inputs, source-gene partition,
implementations, checkpoints, predictions and analyses used in the revised
manuscript. The model is called **BeetleFormer** throughout. The other neural
configurations are architecture comparators or component ablations.

The classification task distinguishes screen-positive iBeetle dsRNAs from
mRNA-derived background controls, following the general dsRNAPredictor
prescreening formulation. The external panel retains the original 20 dsRNA
positives from four publications and 20 background controls. Both primary
sequence isolation and stricter source-gene isolation evaluate the same binary
task and the same 40 external sequences. The 28-fragment dsRIP ranking panel is
not used in the current manuscript or its evaluation pipeline.

Table 1 explicitly separates traditional computational methods from deep-learning
methods. Table 8 and Figure 7 focus on the primary sequence-isolated external
test; the source-gene-isolated sensitivity analysis is summarized in Section 3.5.
Both protocols, their full metrics and all sample-level predictions remain here.


## Reproduce the experiments

This repository contains experimental code and data only. Manuscript/letter
builders, Word and PowerPoint assets, and unrelated exploratory regression
experiments are not included. No sibling authoring folder is required.

Install the dependencies in `requirements.txt`, then run:

```bash
python scripts/verify_release.py
python scripts/reproduce.py --output-dir /absolute/path/to/new_run
# Full training from scratch (choose cpu, cuda or mps):
python scripts/reproduce.py --output-dir /absolute/path/to/new_training --retrain --device cpu
```

The output directory must not exist. Default reproduction reconstructs raw-data
sampling and splits, uses released validation-selected checkpoints and internal
test predictions, recomputes statistics, external predictions, interpretability
and empirical Figures 3–9, and checks prediction consistency. It does not retrain
models. `--retrain` trains all seven primary configurations and the two additional
gene-isolated configurations. `--prepare-only` reconstructs and audits data only.
Training hardware and numerical kernels can affect exact floating-point results.

## Data and protocol

`data/raw/ibeetle` contains the frozen source exports and transcript FASTA.
`provenance/negative_source_provenance.csv` records the transcript, gene and
zero-based half-open coordinates of every sampled background fragment. The
sampling seed is 508. `prepare_data.py` reconstructs the 14,825 positive and
14,825 background sequences and checks exact equality against these records.
Positives satisfy dali11 > 20% or at least one recorded phenotype; missing
measurements are never converted into zero outcomes. The accepted background
lengths are not exactly matched because shorter transcript proposals are rejected.

Source genes in both classes are grouped. Cross-gene groups linked by at least
50% shared distinct 20-mers in either sequence direction are merged, checking
both strands. The split targets 60%/20%/20% of groups using seeds 20260911 and
20260912. The initial assignment is frozen in `provenance/initial_group_assignment.csv`.
The external identities are frozen in `provenance/external_identity_lock.csv`.

The sequence-isolated protocol excludes four modeling records with full
containment or at least 50% of the external sequence's distinct 20-mers shared.
Exact and reverse-complement identity are also checked. The exclusions comprise
one training, two validation and one internal-test record. All 40 external
records are preserved. This retains 29,646 modeling sequences (14,821 positive
and 14,825 background), with 17,957 / 5,606 / 6,083 training/validation/test
sequences. A shared source gene alone is not treated as sequence contamination.

The stricter gene-isolated protocol additionally excludes records assigned to
any of the 37 source genes represented by the original external panel,
including the source genes of its mRNA controls. All 40 fragments map by exact
containment to unique OGS3 source genes. The protocol excludes 47 modeling
records in total and retains 29,603 (14,787 positive and 14,816 background),
with 17,935 / 5,594 / 6,074 training/validation/test sequences. The same initial
partition assignments are retained. No labels, model scores or experimental
magnitudes determine these overlap exclusions. The dataset identity, source
mapping and fixed evaluation plan are recorded in
`provenance/original_external_records.csv`,
`provenance/external_identity_lock.csv`,
`data/raw/original_external/source_manifest.json` and
`config/external_evaluation_plan.json`.

The original external CSV is retained byte-for-byte for traceability. Its old
model scores and heterogeneous experimental readouts are archival fields only.
The current loader exposes the sequence identities, binary labels and source
information; the evaluator recomputes predictions from the selected checkpoints.
Background lengths were sampled from positive lengths, so the two observed
length distributions are not exactly identical despite the matched-length
sampling design. Both span 217–596 bases.

`scripts/audit_original_sources.py` checks all 20 positive records against the
published sequence tables or primer boundaries. The original published source
tables are retained in `data/raw/original_external/source_evidence/`; the report
`provenance/original_external_source_checks.json` records sequence hashes,
source associations and the original primer-trimming rule. Metadata corrections
are documented separately in `provenance/original_external_metadata_notes.json`.
This provenance audit never changes a frozen sequence, label or model input.

`config/experiment.json` defines the common 12-epoch, 128-batch, AdamW protocol.
Both training pools produce 1,692 updates, including a 169-update warmup.
BeetleFormer has 376,226 parameters. Architectures and optimization settings
were fixed before this two-protocol retraining. No external outcome selects
an architecture, epoch or hyperparameter. Seven configurations are trained for
the primary sequence-isolated evaluation; BeetleFormer and the adapted CNN–BiLSTM
are also retrained under gene isolation. To run the same complete training:

```bash
/Users/flash/opt/anaconda3/envs/pyg/bin/python scripts/train_two_protocols.py --device mps
```

One training seed is reported. Gene-bootstrap intervals do not estimate training
variability across seeds. Unused comparator auxiliary heads are retained for
checkpoint and initialization compatibility; they do not contribute to the loss.

The adapted CNN–BiLSTM differs from the released dsRNAPredictor in embedding,
attention aggregation, descriptors and optimization. It must not be described
as the original released tool. `provenance` preserves original preparation and
model-selection source files for traceability; active modules and configuration
are in `scripts`, `model` and `config`.

## Included outputs

- `data/raw/`: frozen iBeetle exports, OGS3 transcripts and original external inputs.
- `provenance/`: sequence coordinates, frozen assignments and source attribution.
- `config/`, `model/`, `utils/`, `scripts/`: experimental implementation.
- `checkpoints/`: selected model weights; `results/`: histories, splits, metrics and predictions.
- `strict_gene/`: additional gene-isolated evaluation inputs, weights and results.
- `figures/`: empirical Figures 3–9 generated from the released results at 600 dpi.

Published DOCX/XLSX source tables in `data/raw/original_external/source_evidence`
are raw experimental evidence, not manuscript-writing files. `python-docx` is
required to read those source tables. `results/manuscript_statistics.json` stores
computed experimental statistics used in the paper; it is not a writing script.
No office suite, PowerPoint, Node.js or image-generation service is required.

`MANIFEST_SHA256.json` records delivered files. `verify_release.py` validates
integrity; rerunning analyses may intentionally update generated outputs.
Source attribution must be retained when redistributing third-party data.

## Sources

iBeetle-Base: Dönitz et al., DOI 10.1093/nar/gku1054. OGS3: Herndon et al.,
DOI 10.1186/s12864-019-6394-6. The original external positives comprise 11
sequences from Gaddelapati et al. (10.1038/s42003-024-06212-7), seven from
Cedden et al. (10.1186/s12915-025-02219-6), one from Knorr et al.
(10.1038/s41598-018-20416-y), and one from Abd El Halim et al.
(10.1038/srep29301). Twenty OGS3 fragments provide the background class.
Source attribution must be retained with third-party inputs. No new biological
experiments are claimed. Background controls are operational classification
labels; they are not experimentally confirmed inactive dsRNAs.

