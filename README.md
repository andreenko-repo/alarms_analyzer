# Alarm Analysis Tool

A tool for demonstration how Python can be used for analyzing industrial control system alarms per **ISA-18.2** and **EEMUA 191** standards. Generates PDF report with recommendations for alarm management improvement.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **EEMUA 191 Health Assessment** - Automatic classification (Very Good → Extremely Dangerous)
- **Bad Actor Analysis** - Pareto identification of alarms contributing to 80% of load
- **Chatter Detection** - Kondaveeti et al method for quantifying rapid cycling
- **Stale & Standing Alarms** - Identify alarms active beyond thresholds or never recovering
- **Avalanche Detection** - Find alarm floods and potential root causes
- **Pattern Mining** - FP-Growth algorithm for co-occurrence analysis
- **Suppression Recommendations** - Association rules for state-based suppression
- **Sequence Analysis** - Markov transition probability matrix

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alarms_analyzer

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
mlxtend>=0.22.0
fpdf2>=2.7.0
```

## Usage

### Basic Usage

```bash
python alarms_analysis_run.py <input_csv> <output_pdf>
```

### Example

```bash
python alarms_analysis_run.py data/synthetic_alarms.csv report.pdf
```

### With Options

```bash
python alarms_analysis_run.py data/synthetic_alarms.csv report.pdf \
    --top_n 20 \
    --window_size 10min \
    --min_support 0.01 \
    --chatter_cutoff 0.05 \
    --stale_hours 24 \
    --lift_threshold 1.5 \
    --verbose
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--top_n` | 15 | Number of top alarms to display in charts |
| `--min_support` | 0.01 | Minimum support threshold for FP-Growth pattern mining |
| `--window_size` | 5min | Time window for pattern analysis |
| `--chatter_cutoff` | 0.05 | Threshold for classifying chattering alarms |
| `--stale_hours` | 24 | Hours before an alarm is considered stale |
| `--repeat_window` | 10 | Minutes window for repeatability analysis |
| `--avalanche_threshold` | 10 | Minimum alarms per minute for avalanche detection |
| `--lift_threshold` | 3.0 | Minimum lift for suppression recommendations |
| `-v, --verbose` | False | Enable debug logging |

## Input Data Format

The tool expects a CSV file with the following structure:

| Column | Index | Description |
|--------|-------|-------------|
| index | 0 | Row index |
| timestamp | 1 | Alarm timestamp (ISO format) |
| tag | 2 | Alarm tag identifier |
| description | 3 | Alarm description |
| condition | 4 | Alarm condition (e.g., HIGH, LOW, FAULT) |

### Condition Format

- **Activation**: `HIGH`, `LOW`, `HIGH HIGH`, `LOW LOW`, `FAULT`, etc.
- **Recovery**: Append `RECOVERED` suffix (e.g., `HIGH RECOVERED`, `LOW LOW RECOVERED`)

### Example CSV

```csv
,timestamp,tag,description,condition
0,2024-01-01 00:01:30,FI106,E-106 REBOILER INLET FLOW,HIGH
1,2024-01-01 00:03:00,FI106,E-106 REBOILER INLET FLOW,HIGH RECOVERED
2,2024-01-01 00:03:30,LI101,T-101 DISTILLATION COLUMN LEVEL,LOW
3,2024-01-01 00:05:06,LI101,T-101 DISTILLATION COLUMN LEVEL,LOW LOW
4,2024-01-01 00:07:00,LI101,T-101 DISTILLATION COLUMN LEVEL,LOW LOW RECOVERED
5,2024-01-01 00:07:30,LI101,T-101 DISTILLATION COLUMN LEVEL,LOW RECOVERED
```

## Report Sections

The generated PDF report contains 13 sections:

1. **Executive Summary** - EEMUA status, flood analysis, key metrics
2. **Temporal Analysis** - Hourly/daily distribution, High Density Alarm Plot (HDAP)
3. **Bad Actor Analysis** - Pareto chart, top frequent alarms
4. **Chatter Analysis** - Kondaveeti chatter index ranking
5. **Repeatability Analysis** - Re-activation frequency within window
6. **Nuisance Identification** - Composite score (chatter + repeatability + frequency)
7. **Stale Alarms** - Alarms active beyond threshold
8. **Standing Alarms** - Alarms that never recover
9. **Avalanche Detection** - Rapid alarm cascades with root cause identification
10. **Pattern Analysis** - FP-Growth co-occurrence patterns
11. **Suppression Recommendations** - Association rule-based suggestions
12. **Sequence Analysis** - Markov transition probability matrix
13. **Recommendations Summary** - Prioritized action items

## EEMUA 191 Benchmarks

The tool classifies alarm system health based on average alarms per 10 minutes:

| Status | Rate (per 10 min) | Description |
|--------|-------------------|-------------|
| 🟢 Excellent | < 1 | Very acceptable |
| 🟡 Good | 1 - 2 | Acceptable |
| 🟠 Warning | 2 - 5 | Manageable |
| 🔴 Critical | 5 - 10 | Likely to miss alarms |
| ⚫ Overloaded | > 10 | Extremely dangerous |

## ISA-18.2 Compliance

The analysis addresses key ISA-18.2 requirements:

- **Section 6.3** - Alarm flood detection (>10 alarms per 10 minutes)
- **Section 11** - Bad actor review and rationalization
- **Section 12** - State-based suppression recommendations
- **Annex B** - Chatter and repeatability metrics

## Project Structure

```
alarm-analysis/
├── alarms_analysis_run.py      # Main analysis script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── synthetic_alarms.csv    # Sample synthetic dataset
│   ├── synthetic_alarms_5_patterns.csv
│   └── test_alarms_original_9.csv
└── report.pdf                  # Example output report
```

## Key Definitions

| Term | Definition |
|------|------------|
| **Bad Actor** | Alarm contributing disproportionately to total alarm load |
| **Chattering** | Alarm repeatedly activating and clearing in rapid succession |
| **Stale Alarm** | Alarm remaining active for extended period (default >24h) |
| **Standing Alarm** | Alarm that activates but never returns to normal |
| **Avalanche** | Many alarms activating in rapid succession (>10/min) |
| **Nuisance Alarm** | Alarm that activates excessively without operator action needed |

## References

- **ISA-18.2-2016** - Management of Alarm Systems for the Process Industries
- **EEMUA 191** - Alarm Systems: A Guide to Design, Management and Procurement
- Kondaveeti, S.R. et al. (2010) - "Quantification of Alarm Chatter Based on Run Length Distributions"

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.