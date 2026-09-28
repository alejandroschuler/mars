"""The reference implementation of the fitting algorithm (the test oracle).

``mars_ref`` implements ``docs/algorithm.md`` literally and slowly, from the
spec alone and with no pymars imports; ``test_reference`` holds its own
sanity tests. The oracle tests import it as ``from reference import
mars_ref``.
"""
