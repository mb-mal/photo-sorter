# PhotoSorter Operations Guide

## Prerequisites
- Install Python 3.10+.
- Install local dependencies: `pip install -r requirements.txt`.
- Ensure Ollama is running locally (`ollama serve`) with the `gemma3:27b` model pulled (`ollama pull gemma3:27b`).
- Place conference images in the source directory (defaults to `img/`).

## Running Classifications
- Dry-run to validate JSON output and database writes without moving files:  
  `python photo_sorter.py --dry-run`
- Full processing to classify, log, and relocate photos:  
  `python photo_sorter.py --destination sorted_photos`
- Reprocess cached photos after updating prompts/settings:  
  `python photo_sorter.py --force`
- Limit batch size for quick iterations:  
  `python photo_sorter.py --limit 25`

## Outputs
- Sorted assets land under the destination root:  
  - `business/<type>/` for contacts, slides, or startup imagery.  
  - `sightseeing/` for non-business scenes.
- SQLite database (`photo_analysis.sqlite` by default) records:
  - Original and destination paths.
  - Category, subtype, and confidence.
  - Exact JSON returned by the model for auditing.

## Monitoring & Maintenance
- Logs stream to stdout; redirect to a file for long runs (`python photo_sorter.py > run.log`).
- On failures, review logged stack traces, adjust prompts, or rename conflicting files.
- Back up the SQLite database regularly for traceability.
- Use `pytest` to run the mocked regression tests once `pytest` is installed:  
  `pytest`
- Schedule recurring runs with `cron` or launchd; include `--force` if re-evaluating evolving classifiers.

## Prompt Refinement Tips
- Capture examples of misclassifications and feed them to the prompt for iterative tuning.
- Tighten JSON schema instructions if the model drifts or adds commentary.
- Experiment with temperature shifts (`--temperature`) to trade precision vs. creativity.
