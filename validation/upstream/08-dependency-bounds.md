# Title: The declared dependency and Python bounds are wrong

## Summary

1. The code passes `ensure_all_finite` to `check_X_y` (`earth.py:229`, `:628`). scikit-learn added that keyword in version 1.6, but the package declares `scikit-learn>=1.0.0`. With scikit-learn 1.5.2, `Earth().fit` fails with `TypeError`.
2. The Python versions disagree. The metadata of the 1.0.4 wheel says `Requires-Python: >=3.9` and lists the classifiers 3.8 to 3.12. HEAD of the tag says `>=3.10`, lists 3.10 to 3.12, and configures ruff for `py39`.
3. `matplotlib>=3.4.0` is a required dependency, and `import pymars` takes 1.2 seconds or more because it imports matplotlib at load time, although fitting does not use it. The plotting functions could import it on first use, or matplotlib could be an optional extra.

## Reproduction

A venv with `scikit-learn==1.5.2` and the 1.0.4 wheel (the pins were `numpy<2.1`, `scikit-learn==1.5.2`, `mars-earth==1.0.4`, Python 3.12):

```python
import numpy as np, sklearn, pymars
print("scikit-learn", sklearn.__version__, "; pymars", pymars.__version__)
rng = np.random.default_rng(0)
X = rng.uniform(size=(50, 3)); y = X[:, 0] + rng.standard_normal(50)
pymars.Earth().fit(X, y)
```

Output (last lines):

```
TypeError: check_X_y() got an unexpected keyword argument 'ensure_all_finite'
```

The import time, in an environment where the wheel is installed:

```python
import sys, time
t = time.perf_counter()
import pymars
print("import pymars: %.2f s; matplotlib loaded: %s" % (time.perf_counter() - t, "matplotlib" in sys.modules))
```

Output with scikit-learn 1.9.1 and numpy 2.5.3 (three runs gave 1.26, 1.20 and 1.21 seconds; the venv with scikit-learn 1.5.2 gave 2.3 seconds):

```
import pymars: 1.20 s; matplotlib loaded: True
```

## Suggestion

Declare `scikit-learn>=1.6` (or use the old keyword name `force_all_finite` through a small shim) and test the lowest bound in CI. Make `requires-python`, the classifiers and the tool targets agree. Move matplotlib to an optional extra, or import it inside the plotting functions. The 2.0 code in the fork declares `scikit-learn>=1.6` and Python 3.10 or later, with the floors explained in `dev/DECISIONS.md`, and it does not depend on matplotlib.
