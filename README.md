# StatisticsForScientists

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/packmad/StatisticsForScientists/publish.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`stats4science` is a lightweight Python package plus companion paper _"From Estimands to Inference: A Practical Tutorial for Robust Statistical Analysis in Scientific Papers"_ for teaching practical statistical inference with an estimand-first perspective.

The repository includes:

- `stats4science/inferential_stats.py`: the main analysis module
- `examples/`: example scripts used to explain how to use this package for different use cases
- `tests/`: unit tests for the statistical helpers and reporting functions
- `latex/estimands_to_inference_tutorial.tex`: the source of the tutorial paper
- `latex/estimands_to_inference_tutorial.pdf`: the compiled PDF version of the paper

> [!warning] 
> It's a Work In Progress!
> It's a new project I've been working on in my spare time; there are still many concepts that need to be presented.
> For example, paired/repeated-measures analyses, regression, contingency tables, ANOVA / mixed models, etc.


## Paper

The tutorial manuscript is available here:

- PDF: [From Estimands to Inference: A Practical Tutorial for Robust Statistical Analysis in Scientific Papers](latex/estimands_to_inference_tutorial.pdf)

## Installation

### From PyPI

```bash
pip install stats4science
```

### For development

See [doc/development.md](doc/development.md) for setup, linting, testing, and release instructions.

## Usage

```python
import stats4science as stats
import numpy as np

group_a = np.array([398, 410, 405, 392, 430])
group_b = np.array([412, 439, 421, 445, 433])

result = stats.compare_independent_groups(group_a, group_b, estimand="mean_difference")
print(stats.report_two_group(result))
```

## Rebuild The Paper

From the `latex/` directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error estimands_to_inference_tutorial.tex
bibtex estimands_to_inference_tutorial
pdflatex -interaction=nonstopmode -halt-on-error estimands_to_inference_tutorial.tex
pdflatex -interaction=nonstopmode -halt-on-error estimands_to_inference_tutorial.tex
```

This produces `latex/estimands_to_inference_tutorial.pdf`.

To reproduce all worked examples from the companion paper:

```bash
python examples/paper_examples.py
# or with options:
python examples/paper_examples.py --show-data
python examples/paper_examples.py --json
```

## Acknowledgments

My warmest thanks to Josh Starmer, the author of [StatQuest](https://www.youtube.com/@statquest) YouTube channel and Tang et al., the authors of the paper [Misuse, Misreporting, Misinterpretation of Statistical Methods in Usable Privacy and Security Papers](https://www.usenix.org/conference/soups2025/presentation/tang).

Your journey continues with these two resources.
