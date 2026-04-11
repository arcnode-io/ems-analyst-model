## Complexity Budget

Default to the simplest implementation that passes the tests.
Before adding any abstraction, pattern, library, or layer — STOP and ask.
Complexity requires explicit approval. Simplicity never does.

If you are about to add a base class, an interface, a factory, a manager, a service layer,
or any indirection that isn't demanded by a failing test — stop. Ask first.


## Decision Gates

STOP and present options before implementing any of the following.
Do NOT implement. Present options and wait for approval.

- Architecture or structural decisions
- Library or framework selection
- Data model design
- Protocol choices
- Anything with physical consequences
- Any decision you are uncertain about


## When Presenting Options

Lead with your recommendation and one sentence why.
Then list alternatives with their tradeoff.

Format:
> I recommend X because Y.
> Alternatives: A (tradeoff), B (tradeoff).

Never present options without a recommendation.
Never present a recommendation without a reason.


## Anti-Bias Rules

| AI Bias | Correct Practice |
|---|---|
| Adds abstraction layers preemptively | YAGNI — build what the test requires, nothing more |
| Presents options without a recommendation | Always lead with recommendation + one sentence why |
| Chains implementation without stopping | Stop at every decision gate and wait for approval |
| Splits files prematurely | 200 line limit, but don't split until you hit it |
| Uses complex patterns to appear thorough | Simple code that passes tests is the goal, not impressive code |
| Makes assumptions when context is missing | Ask. Never assume. |
| Picks a library without presenting alternatives | Always a decision gate — stop and present options |


---


## Implementation Methodology

When presented with a request YOU MUST:

1. Use context7 mcp server or websearch tool to get the latest related documentation. Understand the API deeply and all of its nuances and options.
2. Use TDD: derive expected behavior first, write the failing test, then build until it passes.
3. Start with the simplest happy path test.
4. Think about what the assert should look like.
5. See the test fail.
6. Make the smallest change possible.
7. Check if test passes.
8. Repeat steps 6-7 until it passes.
9. YOU MUST NOT move on until assertions pass.


## Debugging Methodology

### Phase I: Information Gathering
1. Understand the error.
2. Read the relevant source code: try local `.venv`, `node_modules`, or `$HOME/.cargo/registry/src/`.
3. Look at any relevant GitHub issues for the library.

### Phase II: Testing Hypothesis
4. Develop a hypothesis that resolves the root cause. Must only chase root cause solutions. Think hard to decide if it's root cause or NOT.
5. Add debug logs to test hypothesis.
6. If not successful, YOU MUST clean up any artifacts or code attempts in this debug cycle. Then repeat steps 1-5.

### Phase III: Weigh Tradeoffs
7. If successful and fix is straightforward — apply fix.
8. If not straightforward — weigh tradeoffs and provide a recommendation using the options format above.


## Code Structure & Modularity

- **Never break up nested values.** When working with a value that is part of a larger structure, always import or pass the entire parent structure. Never extract or isolate the nested value from its parent context.
- **Get to the root of the problem.** Never write hacky workarounds.
- **Never create a file longer than 200 lines.** If a file approaches this limit, refactor by splitting into modules. Do not split prematurely.
- **Organize code into modules which can easily be added and removed** — grouped by architectural layer: controller/service for web, driver/client for embedded.
- **Strive for symmetry among all projects.** All projects, whatever the language, should follow the same patterns. The only exception is language idioms and idiosyncrasies.
- **Use `cfg.yml` for config variables. NEVER add config vars to env files.**
- **Use `template-secrets.env` to track the list of secrets.**
- **Use environment variables for secrets.** Do NOT conflate secrets with config variables.
- **Use dependency injection for testability.**
- **Keep class names generic:** `TimeseriesClient` not `TimescaleClient`.
- **Use generics judiciously.** If generics don't provide a clear benefit in code reuse, type safety, or API design — use concrete types instead.


## Testing & Reliability

When engaging in TDD:
1. Think about one useful happy path assert.
2. Write the failing test.
3. Write the function with `unimplemented!()` (Rust), `NotImplementedError` (Python), or `throw Error("Not Implemented")` (TypeScript).
4. See the not-implemented error.
5. Make the smallest change until it passes.

- **Use AAA (Arrange, Act, Assert) pattern for all tests.**
- **Unit tests colocated in `src/`.**
- **Integration tests in `tests/`.**
- **Use testcontainers for integration tests** — spin up real databases/services in Docker, session-scoped for performance.
- **Fail fast, fail early.** Detect errors as early as possible and halt. Rely on the runtime to handle the error and provide a stack trace. Do NOT write defensive error handling without a good reason.


## Style

- **Constants:** Top-level declarations in `SCREAMING_SNAKE_CASE`.
- **Use explicit type hints always.** No `Any`.
- **Prefer Pydantic models over dicts for structured data.**
- **Use proper logging, not `print()` debugging.**


## Documentation
 - **Write comments in a terse and casual tone**
- **Comment non-obvious code.** Everything should be understandable to a mid-level developer.
- **Add an inline `# Reason:` comment** for complex logic — explain the why, not the what.
- **Write concise docstrings primarily for an LLM to consume**, secondarily for a document generator.


## AI Behavior Rules

- **Never assume missing context. Ask.**
- **Never hallucinate API or library functions.** Only use known, verified libraries.
- **Never chain steps through a decision gate.** Stop. Present options. Wait.
- **Never declare an API broken without research and confirmation.** If something doesn't work as expected, the first assumption is that you're using it wrong. Before concluding "bug": (1) search docs, forums, and GitHub issues, (2) read the library source, (3) write an isolated probe that eliminates your own usage errors. Only after all three confirm the behavior, label it a bug.



## Python Language Guidelines 🐍
### 🐍 Python-Specific Anti-Bias Rules

- **IGNORE Python's "duck typing" culture - use explicit type hints ALWAYS**
  - NOT: `def process(data):`
  - CORRECT: `def process(data: UnprocessedData) -> List[ProcessedItem]:`
- **No `Any`:**  `Any` typing annotation is NEVER ALLOWED. 
- **Most Python code omits return types - YOU MUST include them**
- **You MUST NOT** use loose types like `dict[str, Any]` or `List[object]`.
- **You MUST ALWAYS** use Pydantic or Pandera models for data structures.
- Name each pydantic or pandera type based on what it represents.
- Define explicit types for all fields in Pydantic or  Pandera models.
- **Training bias toward print() debugging - use proper logging instead**


### Python Testing Guidelines
- **Use actual/expected semantics**  `assert actual == expected`  or `assert_frame_equal(actual_df, expected_df)`

### Python Patterns

- **Prefer structural matching:** Use match/case statements (PEP 636)
- **Prefer validated types:** Use Pydantic or Pandera for type definitions
- **Prefer list comprehensions** for transforming list of objects
- **Use enums** to constrain sets of strings or numbers
- **Add the `Final` typing annotation to all top level SCREAMING_SNAKE_CASE declarations**
- **Use Optional type** for parameters that can be None
- **Write concise Google Style Docstrings for an llm to consume:**

  ```python
  import math
  @dataclass
  class Stats:
    """Statistical metrics for a dataset."""
    mean: float
    median: float
    std_dev: float
    min: float
    max: float

  def calculate_stats(data: list[int]) -> Stats:
    """
    Calculates basic statistics (mean, median, std dev, min, max) for numeric data.

    Args:
        data: List of numbers to analyze

    Returns:
        Stats object with calculated metrics

    Raises:
        ValueError: If data is empty or contains no valid numbers

    Example:
        >>> stats = calculate_stats([1, 2, 3, 4, 5])
        >>> stats.mean
        3.0
    """

    if not data:
        raise ValueError("Data list cannot be empty")

    # Filter out non-finite numbers
    valid = [x for x in data if isinstance(x, (int, float)) and math.isfinite(x)]

    if not valid:
        raise ValueError("No valid numbers found")

    mean = sum(valid) / len(valid)

    sorted_data = sorted(valid)
    n = len(sorted_data)
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2 if n % 2 == 0 else sorted_data[n // 2]

    variance = sum((x - mean) ** 2 for x in valid) / len(valid)
    std_dev = math.sqrt(variance)

    return Stats(
        mean=mean,
        median=median,
        std_dev=std_dev,
        min=min(valid),
        max=max(valid)
    )
  ```


## Python ML Pipeline Guidelines 🤖

### Data Processing & Validation
- **Use Pandera for data validation** with typed DataFrames: `DataFrame[ModelInputSchema]`
- **Use Polars for data processing** over Pandas for performance
- **Implement data transformation pipelines** with clear separation for testability
- **Handle missing data explicitly** - filter null values and validate schema

### Model Training & MLflow
- **Use MLflow for model tracking** and deployment
- **Implement champion/challenger pattern** - compare models and promote best performer
- **Version models automatically** in CI/CD pipeline
- **Use baseline model fallback** when performance degrades

### Data Pipeline Patterns
- **Use scheduled processing** with `schedule` library for daily/periodic runs
- **Define clean model interfaces** with well-defined Pandera types for inputs and outputs
- **Separate data processing from model training** for modularity

### Database & Time Series
- **Use TimescaleDB for time series data** with conflict resolution strategies
- **Implement upsert patterns** - `ON CONFLICT DO UPDATE` for data updates
- **Store processed data with timestamps** in UTC

### Testing Patterns
- **Use fixtures for test data** with proper typing
- **Test data transformations** with expected vs actual DataFrames
- **Use `polars.testing.assert_frame_equal`** for DataFrame comparisons


### Model Development
- **Use Jupyter notebooks for exploration** - `model_development.ipynb`
- **Separate exploration from production code**
- **Implement model comparison logic** 
- **Use statistical validation** for model performance


