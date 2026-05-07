# Contributing to Fusion-pM

Thank you for your interest in contributing to Fusion-pM.

## License of contributions

By submitting a contribution, you agree that your contribution will be licensed under the same license as this repository unless otherwise agreed in writing.

Fusion-pM is distributed for non-commercial research use under the PolyForm Noncommercial License 1.0.0.

## Before contributing

Please do not submit:

- private patient data
- personally identifiable information
- proprietary datasets without permission
- third-party code or data without a compatible license
- confidential clinical or commercial information

## Reporting issues

When reporting a bug, please include:

- operating system
- Python version
- package installation method
- command used
- input format
- full error message
- steps to reproduce

## Pull requests

Before opening a pull request:

1. Create a focused branch.
2. Keep changes small and reviewable.
3. Update documentation if behavior changes.
4. Run basic checks where possible.

Basic syntax check:

```bash
python -m py_compile config.py data_utils.py evaluate.py gradio_app.py model.py test.py train.py train_eval.py
```
