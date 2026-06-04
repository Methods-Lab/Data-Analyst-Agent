"""
EDA Pipeline — Public Step Modules
==================================

This package organises the project's complete data-science pipeline
into seven importable steps that are USED by app.py (not just demos).

    Step 1  →  eda.step_1_loading         data ingestion + first look
    Step 2  →  eda.step_2_cleaning        normalisation + NA handling
    Step 3  →  eda.step_3_exploration     profile, distributions, correlations
    Step 4  →  eda.step_4_preprocessing   encoding + train/test split + target
    Step 5  →  eda.step_5_training        Decision Tree + Random Forest training
    Step 6  →  eda.step_6_evaluation      accuracy, CV, SHAP, importance
    Step 7  →  eda.step_7_prediction      classification / regression / forecast

Each module is fully importable and exposes a small `if __name__ == "__main__":`
demo so you can run it standalone:

    $ python -m eda.step_2_cleaning

The matching Jupyter notebooks in `eda/notebooks/` walk through each step
visually for non-technical readers.
"""

from .step_1_loading       import *  # noqa: F401,F403
from .step_2_cleaning      import *  # noqa: F401,F403
from .step_3_exploration   import *  # noqa: F401,F403
from .step_4_preprocessing import *  # noqa: F401,F403
from .step_5_training      import *  # noqa: F401,F403
from .step_6_evaluation    import *  # noqa: F401,F403
from .step_7_prediction    import *  # noqa: F401,F403
