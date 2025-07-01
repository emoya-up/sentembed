# Motivation :
The Critical Edition of Hannah Ahrendt's Complete Work is being released
[online](https://hannah-arendt-edition.net/home?lang=en).
To help understand the political theorist's work, I propose a framework
compatible with the Gothenburg model and LERA (Pöckelmann et al., 2023).

# Relevant Literature :notebook:
A structural approach for the segmentation of text witnesses showed success
(Frenzel & Stede, 2025).
Molfese et al. (2024) introduce a fully-neural model to allow for
contextualized sentence embeddings.

# Directory Structure

│   readme.md
│   scheduler.lp
│   similarity.py
│   __init__.py
│   
├───compare
│   │   encode.py
│   │
│   └───__pycache__
│           encode.cpython-312.pyc
│           encode.cpython-313.pyc
│
├───data
│       embeddings_en.json
│       embeddings_es.json
│       embeddings_fr.json
│       embeddings_ru.json
│       embeddings_zh.json
│       mask.json
│       sent_speeches_all.csv
│       sent_text_all.csv
│       UNPD_en.txt
│       UNPD_es.txt
│       UNPD_fr.txt
│       UNPD_ru.txt
│
├───embeddings
│   │   labse.py
│   │   laser.py
│   │   __init__.py
│   │
│   └───__pycache__
│           labse.cpython-312.pyc
│           labse.cpython-313.pyc
│           similarity.cpython-312.pyc
│           __init__.cpython-312.pyc
│           __init__.cpython-313.pyc
│
├───preprocessing
│       analyze.py
│       preprocessing.py

# References
Frenzel, S., & Stede, M. (2025). Sentence-Alignment in Semi-parallel Datasets. In A. Kazantseva, S. Szpakowicz, S. Degaetano-Ortlieb, Y. Bizzoni, & J. Pagel (Eds.), Proceedings of the 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025) (pp. 87–96). Association for Computational Linguistics. (https://aclanthology.org/2025.latechclfl-1.9/)
Molfese, F., Bejgu, A., Tedeschi, S., Conia, S., & Navigli, R. (2024). CroCoAlign: A Cross-Lingual, Context-Aware and Fully-Neural Sentence Alignment System for Long Texts (Y. Graham & M. Purver, Eds.; pp. 2209–2220). Association for Computational Linguistics. (https://aclanthology.org/2024.eacl-long.135/)
Pöckelmann, M., Medek, A., Ritter, J., & Molitor, P. (2023). LERA—an interactive platform for synoptical representations of multiple text witnesses. Digital Scholarship in the Humanities, 38(1), 330–346. (https://doi.org/10.1093/llc/fqac021)
