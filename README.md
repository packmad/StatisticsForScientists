# StatisticsForScientists

`StatisticsForScientists` is a small Python project plus companion paper _"From Estimands to Inference: A Practical Tutorial for Robust Statistical Analysis in Scientific Papers"_ for teaching practical statistical inference with an estimand-first perspective.

The repository includes:

- `inferential_stats.py`: the main lightweight analysis module
- `examples.py`: reproducible worked examples used in the paper
- `tests/`: unit tests for the statistical helpers and reporting functions
- `latex/estimands_to_inference_tutorial.tex`: the source of the tutorial paper
- `latex/estimands_to_inference_tutorial.pdf`: the compiled PDF version of the paper

## Paper

The tutorial manuscript is available here:

- PDF: [From Estimands to Inference: A Practical Tutorial for Robust Statistical Analysis in Scientific Papers](latex/estimands_to_inference_tutorial.pdf)

## Installation

Create or activate your environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run The Examples

The paper's worked examples are implemented in `examples.py`. 
Please refer to this file for guidalines on how to use module `inferential_stats.py`.

```bash
python3 examples.py
```

Useful options:

- `python examples.py --show-data` prints the raw datasets used in the paper
- `python examples.py --json` emits machine-readable results for all worked examples

## Run The Tests

```bash
python3 -m unittest discover -s tests
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


## Acknowledgments

My warmest thanks to Josh Starmer, the author of [StatQuest](https://www.youtube.com/@statquest) YouTube channel and Tang et al., the authors of the paper [Misuse, Misreporting, Misinterpretation of Statistical Methods in Usable Privacy and Security Papers](https://www.usenix.org/conference/soups2025/presentation/tang). 

Your journey continues with these two resources.