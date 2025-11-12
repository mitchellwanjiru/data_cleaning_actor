# data_cleaning_actor

A Python-based Apify Actor for automatic data cleaning, summarization, and visualization of CSV or Excel files.

## Features

- Upload CSV or Excel files
- Clean data (remove missing, standardize types)
- Compute summary statistics (mean, median, missing values)
- Generate correlation matrix and plots (histogram, heatmap)
- Outputs cleaned CSV, JSON summary, and optional plot image

## Usage

### Inputs

- `dataFile`: Path to CSV or Excel file to process
- `columnsToAnalyze`: (optional) List of columns to include in analysis
- `summaryType`: "stats", "correlations", or "charts"

Example `key_value_stores/default/INPUT.json`:

```json
{
  "dataFile": "sample.csv",
  "columnsToAnalyze": [],
  "summaryType": "stats"
}
```

### Outputs

- `key_value_stores/default/cleaned.csv`: Cleaned data file
- `key_value_stores/default/summary.json`: JSON summary (mean, median, missing values, correlations)
- `key_value_stores/default/plot.png`: Plot image (if requested)

## Running Locally

1. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Prepare your input file and `INPUT.json` as above.
3. Run the Actor:
   ```powershell
   python src/main.py
   ```
4. Check outputs in `key_value_stores/default/`

## Publishing to Apify

1. Ensure all required files are present:
   - `.actor/actor.json`, `input_schema.json`, `output_schema.json`, `dataset_schema.json`
   - `src/main.py`, `requirements.txt`, `Dockerfile`, `AGENTS.md`, `README.md`
2. Test with real data and input options.
3. Log in and push:
   ```powershell
   apify login
   apify push
   ```

## Notes

- Only numeric columns are used for correlations and plots.
- All NaN values in JSON output are converted to `null` for compatibility.
- For questions or issues, see AGENTS.md for Apify Actor guidance.
