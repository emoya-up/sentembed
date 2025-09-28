# Motivation :
The Critical Edition of Hannah Ahrendt's Complete Work is being released
[online](https://hannah-arendt-edition.net/home?lang=en).
To help understand the political theorist's work, I propose a framework
compatible with the Gothenburg model and LERA (Pöckelmann et al., 2023).

# How to run :
The core functionality can be executed with uv or inline.

#### For uv:
Run these commands from the root directory of this repository. By default, this compares two versions of Hannah Arendt's homage to Franz Kafka.
``` bash
uv sync
uv run -m compare.similarity
```
See [the uv documentation](https://docs.astral.sh/uv/getting-started/) to install this package manager.

#### Inline with Python and relevant dependencies:
If you prefer not to install uv, run this set of commands with pip.
``` bash
pip install -r requirements.txt
python3 -m compare.similarity
```

The executed script returns a list of average cosine similarities for all language pairs in a pandas DataFrame.

# Relevant Literature :
A structural approach for the segmentation and alignment of comparable texts showed good results (Frenzel & Stede, 2025).
Molfese et al. (2024) introduce a fully-neural model to allow for
contextualized sentence embeddings.

# Directory Structure
``` diff
# .
  ├── __init__.py
+ ├── compare
  │   ├── __init__.py
  │   ├── context.py
  │   ├── similarity.py
  │   └── vis.py
+ ├── data
+ ├── embeddings
  │   ├── __init__.py
  │   ├── labse.py
  │   └── laser.py
  ├── main.py
+ ├── preprocessing
  │   ├── __init__.py
  │   ├── align_kafka.py
  │   ├── analyze.py
  │   ├── preprocessing.py
  │   └── write_parallel.py
# ├── pyproject.toml
  ├── readme.md
  ├── scheduler.lp
# └── uv.lock
```

# References
Frenzel, S., & Stede, M. (2025). Sentence-Alignment in Semi-parallel Datasets. In A. Kazantseva, S. Szpakowicz, S. Degaetano-Ortlieb, Y. Bizzoni, & J. Pagel (Eds.), Proceedings of the 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025) (pp. 87–96). Association for Computational Linguistics. (https://aclanthology.org/2025.latechclfl-1.9/)
Molfese, F., Bejgu, A., Tedeschi, S., Conia, S., & Navigli, R. (2024). CroCoAlign: A Cross-Lingual, Context-Aware and Fully-Neural Sentence Alignment System for Long Texts (Y. Graham & M. Purver, Eds.; pp. 2209–2220). Association for Computational Linguistics. (https://aclanthology.org/2024.eacl-long.135/)
Pöckelmann, M., Medek, A., Ritter, J., & Molitor, P. (2023). LERA—an interactive platform for synoptical representations of multiple text witnesses. Digital Scholarship in the Humanities, 38(1), 330–346. (https://doi.org/10.1093/llc/fqac021)
