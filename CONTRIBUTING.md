# Contributing

Thanks for your interest in fast-matrix-rank. Bug reports, benchmark results and pull requests are welcome.

## Reporting a problem

Open an issue and include:

- the matrix type, size and sparsity (or a small code example that builds it)
- the rank you expected and the rank you got
- your Python, NumPy and SciPy versions and your operating system

Examples of matrices where the LU path gives the wrong rank are especially useful.

## Running the tests

```bash
git clone https://github.com/Jaypatel2611/fast-matrix-rank.git
cd fast-matrix-rank
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest
```

## Pull requests

- Keep each pull request focused on one change.
- Add a test for every bug fix and every new behaviour.
- Make sure `pytest` passes before you open the pull request.
- If you change performance-related code, run `python benchmark.py` and include your results (with your CPU, OS and package versions) in the description.

## AI-assisted contributions

If you used an AI tool to write any of your code or text, say so in the pull request, and make sure you understand and can explain what you submit.

## License

By contributing, you agree that your contribution is licensed under the MIT License of this project.
