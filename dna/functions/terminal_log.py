"""Terminal (stdout) logging of the intermediate text a run generates.

Prints the grounded objects, per-frame descriptions, and final progress
prediction in a readable, sectioned format. Used by both ``DNA`` and
``Experimental`` when ``verbose >= 1``.
"""

import numpy as np

_RULE = "─" * 72


def _section(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def log_run(
    task: str,
    *,
    objects=None,
    description=None,
    success_criteria=None,
    progress=None,
    failure=None,
    feedback=None,
) -> None:
    """Print whatever intermediate outputs are available for one run.

    Args:
        task: The task description.
        objects: Grounded object list (dna method only); skipped if None.
        description: Mapping of frame index -> description string, or a list of
            description strings; skipped if None.
        success_criteria: The model's explicit definition of full task completion
            (the prompt's "completion state" field); skipped if None.
        progress: Final per-frame progress array in [0, 1]; skipped if None.
        failure: Failure description (dna_feedback only); skipped if falsy.
        feedback: Corrective feedback (dna_feedback only); skipped if falsy.
    """
    _section(f"[dna] task: {task!r}")

    if objects is not None:
        print(f"\nobjects ({len(objects)}): {objects}")

    if description is not None:
        items = description.items() if isinstance(description, dict) else enumerate(description)
        print("\ndescriptions:")
        for i, desc in items:
            print(f"  [{i}] {desc}")

    if success_criteria is not None:
        print(f"\nsuccess criteria: {success_criteria}")

    if progress is not None:
        prog = np.asarray(progress, dtype=float).reshape(-1)
        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(f"\nprediction ({len(prog)} frames): {prog}")
        if len(prog):
            print(f"final: {prog[-1]:.3f}   max: {prog.max():.3f}")

    if failure:
        print(f"\nfailure: {failure}")
    if feedback:
        print(f"feedback: {feedback}")
    print()
