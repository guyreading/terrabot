# Post reports as comments in the curent commit
echo "## Param Diffs From Last Exp" >> report.md
dvc params diff --show-md >> report.md

echo "## Metric Diffs from Last Exp" >> report.md
dvc metrics diff --show-md >> report.md

echo "## Model Plots" >> report.md
cml-publish  "data/faction-picker-bot/metrics/alchemists chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/auren chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/chaosmagicians chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/cultists chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/darklings chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/dwarves chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/engineers chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/fakirs chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/giants chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/halflings chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/mermaids chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/nomads chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/swarmlings chart.png" --md >> report.md
cml-publish  "data/faction-picker-bot/metrics/witches chart.png" --md >> report.md