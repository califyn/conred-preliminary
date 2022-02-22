Here is my code for ConRED, a **con**trastive **R**NA-**e**xpression and pathological **d**iagnosis matcher.

This is a preliminary draft.
Research is ongoing.

Code is primarily in these three files (all in the ``code`` directory):
- ``train.py`` for actual training code
- ``datasets.py`` for dataset/data-utils generation
- ``model.py`` for initializing custom models

Data is not in this repository as it's a few gigabytes too big.

Also, scripts for generating figures are in ``figures_script.py``.

# Davidson Fellowship Addendum

Parts that were written by me:
- ``alt.py``
- ``colscript.py``
- ``datasets.py``
- ``figures_script.py``
- ``final_script.py``
- ``load_model.py``
- ``model.py``
- ``run_script.py``
- ``train.py`` EXCEPT for lines 111 to 164

Parts that were NOT written by me:
- ``biobert/*`` (from BioBERT)
- ``tabtransformer/*`` (from TabTransformer)
- ``sync_batchnorm/*`` (from Sync Batchnorm)
- ``train.py``, lines 111 to 164, written by my mentor
