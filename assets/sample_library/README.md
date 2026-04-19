Sample library layout:

- `golden_set/rvc_sample/`
  - Migrated baseline samples from the legacy `RVC Sample` folder.
- `review_set/`
  - Project-owned copies of representative and known-failure samples from `book-res-studio`.
- `manifest.jsonl`
  - Full structured sample inventory.
- `review_candidates.json`
  - Bounded subset for manual review and labeling.
- `labels/manual_labels.jsonl`
  - Manual confirmations written by the labeler UI.

Current seed sources:

- `/Users/yanwei/Downloads/RVC Sample`
- `/Users/yanwei/dev/namibox/book-res-studio/data/library/book-res/tape3a_002013`
- `/Users/yanwei/dev/namibox/book-res-studio/data/library/book-res/tape3a_002011`
- `outputs/audio_analysis/book_res_full_20260418/audio_features.sqlite`

To rebuild:

```bash
uv run python tools/build_sample_library.py --max-review-items 96
```

To run the labeler:

```bash
uv run python tools/sample_library_labeler.py
```
