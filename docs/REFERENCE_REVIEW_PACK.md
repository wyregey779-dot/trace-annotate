# Reference boundary review pack

`scripts/build_reference_review_pack.py` turns an existing `trace-annotate.reference-candidates.v1` document into a new, read-only evidence directory. It verifies the video and frame-map hashes, checks every decoded frame PTS against the supplied map, and extracts the exact frame before and at each available boundary. The source video must be available locally, with `ffmpeg` and `ffprobe` on `PATH`.

```powershell
python scripts/build_reference_review_pack.py --reference D:\path\to\reference.json --out D:\path\to\new-review-pack
```

The output directory must not already exist. Open `index.html` locally to inspect adjacent frames, source hashes, decoded-video PTS and time base, annotation-timeline seconds, provenance, declared kind, and reasons a candidate is ineligible. The page identifies whether the decoded video is the original source or a prepared playback copy. Exact time fractions appear alongside rounded, human-readable values; video PTS and the annotation timeline can have different origins. `manifest.json` lists the SHA-256 of the HTML, stylesheet, and each PNG. The page uses no JavaScript or network resources. A failure after output creation retains `FAILED.json` and incomplete files; pre-creation input validation can fail without creating a directory or `FAILED.json`. A pack without a complete `manifest.json` is not usable as evidence.

The page is **not a review decision interface**. Every event stays unreviewed, and these still frames cannot establish audio content, gradual transitions, annotation correctness, human adjudication, or holdout eligibility. Review the original video and separately document and sign off any human decision. Do not treat screenshots or this pack as signed ground truth.
