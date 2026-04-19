# RVC Automatic Conversion MECE Design

## 1. Purpose

This document defines a system-level design for large-scale RVC-based voice conversion over the `book-res` corpus and similar audio sets.

The goal is not to maximize conversion coverage at all costs. The goal is to:

- convert the dominant easy cases well;
- isolate complex cases into dedicated routes;
- avoid producing obviously broken output at scale;
- provide deterministic routing, structured metadata, and automatic fallback;
- support batch processing over hundreds of thousands of files without relying on manual listening.

This design is based on the full-dataset analysis in:

- `/Users/yanwei/src/Retrieval-based-Voice-Conversion-WebUI/outputs/audio_analysis/book_res_full_20260418/summary.md`
- `/Users/yanwei/src/Retrieval-based-Voice-Conversion-WebUI/outputs/audio_analysis/book_res_full_20260418/summary.json`


## 2. Dataset Facts That Drive The Design

The architecture must follow the observed distribution rather than intuition.

### 2.1 Core distribution

- Total analyzed files: `132172`
- Total errors: `1`
- `track_audio`: `129409`
- `unit_audio`: `2763`
- Duration median: `2.509s`
- Duration p75: `5.096s`
- Duration p90: `17.426s`
- Duration p95: `37.186s`
- Duration p99: `177.229s`
- Longest file: `3425.033s`

### 2.2 Duration buckets

- `<1s`: `22529`
- `1-3s`: `53008`
- `3-8s`: `32998`
- `8-25s`: `13938`
- `25-60s`: `5752`
- `60-180s`: `2638`
- `>180s`: `1309`

### 2.3 Light classification distribution

- `music_or_bgm_candidate`: `56630`
- `clean_or_mixed_speech_candidate`: `31119`
- `clean_speech_candidate`: `31118`
- `short_speech_or_sfx`: `7998`
- `low_confidence`: `5189`
- `fragmented_or_sfx_candidate`: `117`
- `invalid_or_silent`: `1`

### 2.4 Main implications

1. The default path must optimize for short speech, not long audio.
2. Global fixed thresholds like `0.75s`, `8s`, and `25s` are not acceptable as universal rules.
3. Background-music handling matters, but must not be applied blindly.
4. Extremely short audio is common and often valid, not automatically anomalous.
5. Long mixed content is important, but is not the dominant global case and should be isolated into its own route.


## 3. Design Goals

### 3.1 Functional goals

- Choose one and only one processing route for each input file.
- Support same-direction voice replacement by default:
  - male input -> male model
  - female input -> female model
- Preserve short SFX events instead of converting them into artifacts.
- Separate continuous BGM only when justified.
- Provide a dedicated route for long mixed content.

### 3.2 Quality and safety goals

- No silent corruption.
- No hidden passthroughs without metadata.
- No UVR on clean speech by default.
- No isolated conversion of unstable micro-segments.
- Any low-confidence or failed output must fall back to original audio.

### 3.3 Operational goals

- Deterministic classification and routing.
- Batch-friendly quality checks.
- Fully structured metadata for later analysis.
- Route-level and model-level monitoring.


## 4. MECE Audio Type System

Every file must land in exactly one type.

### 4.1 `invalid_or_silent`

Definition:

- file missing, unreadable, or decode failure;
- near-silent audio;
- no meaningful speech and no meaningful non-speech content.

Default action:

- `processing_mode = passthrough_original`
- `fallback_reason = invalid_or_silent`

### 4.2 `short_voice`

Definition:

- speech-dominant short audio;
- no continuous BGM bed;
- no dominant transient SFX problem;
- one speaker or one dominant speaker.

Default action:

- `processing_mode = single_voice`

### 4.3 `short_voice_with_sfx`

Definition:

- speech-dominant short audio;
- includes sparse transient SFX such as chimes, clicks, prompt tones, page sounds, or other short non-speech events;
- these events do not form continuous accompaniment.

Default action:

- `processing_mode = speech_sfx_segments`

### 4.4 `voice_with_bgm`

Definition:

- speech present;
- continuous background music or accompaniment present;
- music exists as a time-continuous bed, not just sparse events.

Default action:

- `processing_mode = separate_bgm_voice`

### 4.5 `long_mixed`

Definition:

- long audio;
- or multiple speakers / male-female alternation;
- or long-form audio with music and structural complexity;
- or lesson-style / dialogue-style mixed content.

Default action:

- `processing_mode = long_mixed_pipeline`

### 4.6 `low_confidence`

Definition:

- classification evidence is contradictory;
- speaker/gender routing is unstable;
- segmentation cannot be resolved safely;
- UVR or output validation fails;
- quality gate detects serious risk.

Default action:

- `processing_mode = passthrough_original`
- `fallback_reason = low_confidence`


## 5. End-To-End Decision Flow

```text
input
-> decode gate
-> invalid_or_silent ? yes -> passthrough_original

-> light feature extraction
-> continuous BGM likely ?
   -> yes:
      -> long/complex ? yes -> long_mixed_pipeline
      -> no -> separate_bgm_voice

-> sparse transient SFX likely ?
   -> yes -> speech_sfx_segments

-> short speech-dominant and simple ?
   -> yes -> single_voice

-> long/multi-speaker/structurally complex ?
   -> yes -> long_mixed_pipeline

-> otherwise -> passthrough_original (low_confidence)
```

Key constraint:

- execution code must not rewrite the classification after routing;
- fallback is handled only via explicit validation and metadata.


## 6. Feature Layers

## 6.1 Light features

Used for all files.

Inputs:

- duration;
- RMS / peak / silence ratio;
- voiced coverage;
- split segment count;
- short-segment ratio;
- harmonic ratio;
- percussive ratio;
- onset strength;
- spectral flatness.

Purpose:

- coarse routing;
- candidate generation;
- identifying files that need deep analysis.

## 6.2 Deep features

Used for boundary and complex cases only.

Inputs:

- stronger VAD;
- F0 summary and voiced frame statistics;
- speaker embedding or MFCC embedding clustering;
- transient SFX detection;
- continuity of non-speech energy;
- UVR before/after validation.

Purpose:

- stable segment routing;
- speaker/gender confidence;
- SFX preservation;
- complex-route confirmation.


## 7. Threshold Design

Thresholds must come from three layers.

## 7.1 Layer A: global guard rails

These are hard safety bounds.

Examples:

- `very_short_floor`
- `short_voice_upper` safe range
- `long_audio_min` safe range
- confidence lower bounds

These remain configurable but not fully free.

## 7.2 Layer B: dataset-adaptive thresholds

Derived from corpus statistics.

Using the current corpus:

- `p05 = 0.706s`
- `p50 = 2.509s`
- `p75 = 5.096s`
- `p90 = 17.426s`
- `p95 = 37.186s`

Recommended usage:

- `short_voice_upper` derived from `p75`
- `long_audio_min` derived from `p90`
- `very_short_risk` derived from `p05`

These replace fixed global cutoffs like `0.75`, `8`, and `25`.

## 7.3 Layer C: file-local adaptive thresholds

Used mainly by the segmenter.

Examples:

- short-segment threshold derived from local voiced segment median;
- merge-gap threshold derived from local pause distribution;
- word-cluster gap derived from lower local pause quantiles.

Interpretation:

- global thresholds choose the route;
- local thresholds choose segment boundaries and merges.


## 8. Gender And Speaker Routing

## 8.1 Non-goal

The system must not use a single F0 threshold as the entire gender-routing mechanism.

That approach is too fragile for:

- child voices;
- unstable voiced frames;
- short words or letters;
- breathy or expressive speech;
- mixed Chinese and English short utterances.

## 8.2 Recommended routing evidence

Each segment should use weighted evidence from:

- pitch evidence:
  - F0 median;
  - voiced frame count;
  - F0 stability;
- speaker embedding evidence:
  - same-cluster continuity;
  - cluster stability;
- duration confidence:
  - very short segments get lower confidence.

Output fields:

- `gender_route = male | female | unknown`
- `gender_confidence = 0..1`
- `speaker_cluster_id`

Low-confidence segments must not be forcibly routed.


## 9. Segmentation And Merge Rules

## 9.1 Principles

- segment only when necessary;
- do not optimize for equal-length segments;
- do not convert unstable micro-segments independently;
- SFX must be protected explicitly, not implicitly.

## 9.2 Segmentation pipeline

1. Use VAD to obtain speech candidates.
2. Refine boundaries using silence valleys and spectral changes.
3. Extract segment-level embeddings, F0, and transient features.
4. Run merge logic before any conversion.

## 9.3 Merge rules

### Rule A: same-cluster merge

Merge adjacent segments if:

- speaker embedding is close;
- gender route is consistent or non-conflicting;
- pause gap is short.

### Rule B: word-cluster merge

Used for textbook-style reading:

- short words;
- letters;
- phrase-by-phrase reading;
- repeated short pauses.

If several adjacent short segments share speaker identity and short gaps, they should be merged into a reading cluster.

### Rule C: ultra-short absorb

If a segment is too short for stable conversion:

- first try to absorb into previous same-cluster segment;
- otherwise absorb into next same-cluster segment;
- otherwise mark as preserve/fallback;
- do not send it alone to RVC.


## 10. SFX Preservation

Short transient non-speech events should be tagged as `sfx_event` when they show:

- extremely short duration;
- no stable F0;
- sharp spectral profile;
- low speech likelihood;
- sparse occurrence pattern.

Once tagged:

- they must not be routed through RVC;
- they must not be merged into speech by default;
- they must be preserved on the original timeline.

This is how prompt tones, page sounds, and similar events remain natural.


## 11. UVR / Vocal Separation Policy

## 11.1 Default rule

UVR is not a default first step.

It is used only for:

- `voice_with_bgm`
- `long_mixed`

It is not used for:

- `short_voice`
- `short_voice_with_sfx`

## 11.2 Continuous BGM test

The classifier must distinguish:

- continuous accompaniment;
- sparse transient SFX.

Evidence for continuous BGM includes:

- sustained non-speech energy;
- harmonic/percussive co-presence over time;
- non-speech regions staying acoustically active;
- repeated onset structure across longer spans.

## 11.3 UVR post-validation

After separation, compare before/after features:

- voiced coverage;
- speech energy;
- non-speech leakage;
- total duration consistency.

If the separated vocal track loses too much speech structure:

- `uvr_validation_failed = true`
- route must fallback rather than continue blindly.


## 12. Processing Modes

The system should standardize on five execution modes:

- `passthrough_original`
- `single_voice`
- `speech_sfx_segments`
- `separate_bgm_voice`
- `long_mixed_pipeline`

Avoid keeping overlapping legacy names such as `safe_long` once the new routing is in place.


## 13. Model Routing And Parameter Profiles

## 13.1 Model pools

Keep a small explicit set of models:

- `male_primary`
- `female_primary`
- `male_alt`
- `female_alt`

Do not let automatic routing choose from a large uncontrolled model pool in the first version.

## 13.2 Default routing

- `short_voice`: dominant gender route -> primary gender model
- `short_voice_with_sfx`: same as above, speech only
- `voice_with_bgm`: separated vocal route by dominant gender or segment route
- `long_mixed`: segment route by cluster + gender
- `low_confidence`: no conversion

## 13.3 Parameter profiles

Parameters must live in named profiles, not scattered constants.

Each profile should define:

- `f0_method`
- `f0_up_key`
- `index_rate`
- `protect`
- `rms_mix_rate`
- `filter_radius`
- `resample_sr`

Suggested profile groups:

- `default_short_voice`
- `default_speech_sfx`
- `default_bgm_voice`
- `default_long_mixed`


## 14. Fallback Policy

Fallback is mandatory.

Fallback to original audio if any of the following holds:

- decode failure;
- invalid or near-silent input;
- low-confidence classification;
- unresolved speaker/gender conflict;
- UVR validation failure;
- conversion exception;
- output quality gate failure;
- strong duration mismatch;
- strong voiced-coverage collapse;
- clipping or audio collapse.

The job should still finish with explicit metadata:

- `status = fallback`
- `fallback_used = true`
- `fallback_reason = ...`


## 15. Automatic Quality Gate

## 15.1 File-level checks

Compare input vs output:

- duration delta ratio;
- RMS / loudness delta;
- voiced coverage delta;
- silence ratio delta;
- clipping;
- severe energy collapse;
- severe speech loss.

Example failure signals:

- duration mismatch above a safe limit;
- output almost loses all speech;
- output is mostly silence;
- obvious clipping or waveform collapse.

## 15.2 Batch-level checks

Aggregate by:

- audio type;
- processing mode;
- model;
- book / tape / directory;
- fallback rate;
- UVR failure rate;
- low-confidence rate;
- mean duration delta;
- mean loudness delta.

If a mode, model, or book segment spikes in failure rate, that route should be paused for review.


## 16. Metadata Schema

Each processed file should emit a unified metadata record such as `job_metadata.json`.

Required top-level fields:

- `job_id`
- `input_path`
- `output_path`
- `status`
- `audio_summary`
- `classification`
- `selected_plan`
- `speaker_analysis`
- `segments`
- `conversion`
- `quality_gate`
- `timing`

Each segment must include:

- `segment_id`
- `start`
- `end`
- `duration`
- `segment_type`
- `speaker_cluster_id`
- `gender_route`
- `gender_confidence`
- `merge_reason`
- `action`

This metadata is the basis for:

- debugging;
- auditability;
- threshold calibration;
- batch QA;
- later model and strategy optimization.


## 17. Recommended Code Architecture

Recommended modules:

- `tools/rvc_pipeline/audio_features.py`
- `tools/rvc_pipeline/thresholds.py`
- `tools/rvc_pipeline/classifier.py`
- `tools/rvc_pipeline/segmenter.py`
- `tools/rvc_pipeline/sfx_detector.py`
- `tools/rvc_pipeline/speaker_router.py`
- `tools/rvc_pipeline/uvr_router.py`
- `tools/rvc_pipeline/executor.py`
- `tools/rvc_pipeline/quality_gate.py`
- `tools/rvc_pipeline/metadata.py`
- `tools/rvc_pipeline/profiles.py`

Responsibilities:

- `audio_features.py`: extract features only
- `thresholds.py`: build global/dataset/local thresholds
- `classifier.py`: produce MECE class and confidence
- `segmenter.py`: segment and merge
- `sfx_detector.py`: detect and preserve sparse events
- `speaker_router.py`: cluster, gender route, model route
- `uvr_router.py`: decide and validate separation
- `executor.py`: execute chosen processing mode
- `quality_gate.py`: accept or fallback
- `metadata.py`: serialize structured outputs


## 18. Integration With Existing Files

### 18.1 `tools/rvc_auto_convert.py`

Future role:

- orchestration only;
- no embedded route-specific heuristics;
- no hidden policy rewrites.

### 18.2 `tools/process_mixed_long_audio.py`

Future role:

- dedicated implementation of `long_mixed_pipeline`;
- no global first-pass classification.

### 18.3 `tools/simple_rvc_flask.py`

Future role:

- UI and task control only;
- display classification, selected plan, quality results, and metadata;
- optional explicit user override, but no implicit hidden re-routing in the UI.


## 19. Implementation Priority

### Phase 1

Implement first:

- classifier
- thresholds
- quality gate
- metadata schema

Reason:

- this locks down routing correctness and safe fallback.

### Phase 2

Implement next:

- segmenter
- SFX detector
- speaker router

Reason:

- these improve the most common short-speech paths.

### Phase 3

Implement last:

- UVR router
- long mixed pipeline refinement

Reason:

- these are important but should not dominate the initial architecture.


## 20. Test Strategy

### 20.1 Unit tests

- invalid/silent detection
- short voice -> `single_voice`
- short voice with sparse SFX -> `speech_sfx_segments`
- continuous BGM -> `separate_bgm_voice`
- long complex file -> `long_mixed_pipeline`
- ultra-short speech segment absorb
- SFX preservation
- low-confidence fallback

### 20.2 Integration tests

Use known samples covering:

- short male speech
- short female speech
- mixed male/female short speech
- speech with BGM
- song-like input
- long mixed lesson-style audio
- textbook-style short-word reading

### 20.3 Batch regression

Use the current `samples/` outputs from the corpus analysis to validate:

- class distribution drift;
- fallback rate drift;
- UVR route stability;
- model-route stability.


## 21. Optional Use Of Local LLM

Local `ollama + gemma4:26b` may be useful, but only in auxiliary roles.

Recommended uses:

- explain low-confidence or fallback decisions for human review;
- summarize batch-quality anomalies;
- help cluster failure patterns across metadata logs.

Do not use the local LLM as:

- the primary classifier;
- the primary gender-routing mechanism;
- the primary UVR decision engine.

The main processing route must remain deterministic, measurable, and fast.


## 22. Immediate Next Step

The next implementation step should not add more heuristics directly into the current entrypoints.

The next step should be:

1. formalize classifier outputs and metadata schema;
2. formalize adaptive threshold sources;
3. separate routing from execution;
4. add quality-gate fallback before expanding more conversion logic.
