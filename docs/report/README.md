# AeroRozum hackathon experimental report

Four-page scientific report aligned with AfterQuery: Break the Agent. No literature review. The report distinguishes mission reward, image-level perception metrics, numerical checks, infrastructure errors, and incomplete work.

- `AeroRozum-hackathon-report.pdf`: compiled paper.
- `AeroRozum-hackathon-report.tex` and `tables/`: editable LaTeX source.
- `experiment-ledger.csv`: 300 recorded rows, including shared parents/phases and unexecuted follow-ups. These are **not 300 independent experiments**.
- `evidence-manifest.json`: SHA-256 hashes of the input snapshots used for the generated tables and ledger.

Compile from this directory with `pdflatex -interaction=nonstopmode -halt-on-error AeroRozum-hackathon-report.tex` twice. TeX Live base/recommended suffices. From the repository root, regenerate tables and the ledger with `python3 scripts/build_hackathon_report.py`.

The source archive preserves repository-relative paths, the generator, and the hashed input snapshots. It does not contain raw flight videos, credentials, or model weights. No experiments were restarted to produce this report. Scientific status is the recorded hackathon wrap-up snapshot; the independent domain-expansion physics solve was allowed to finish later in the background.
