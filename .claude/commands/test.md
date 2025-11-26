Run the test suite for VERA-MH with comprehensive options:

1. Check if tests directory exists, suggest running /setup-dev if not
2. Check if pytest and pytest-cov are installed
3. Parse command arguments:
   - `--unit`: Run only unit tests (fast)
   - `--integration`: Run only integration tests
   - `--e2e`: Run only E2E tests (slow)
   - `--no-cov`: Skip coverage reporting
   - `[path]`: Run tests in specific file or directory
4. Run pytest with appropriate markers and coverage
5. Display test results with pass/fail counts
6. Show coverage report with color-coded feedback:
   - Red: <50% coverage (needs attention)
   - Yellow: 50-75% coverage (approaching target)
   - Green: >75% coverage (meets target)
7. If coverage below threshold, suggest specific files to add tests for
8. If tests fail, show failure details and suggest fixes
9. Remind about test pyramid structure (unit > integration > e2e)

Examples:
- `/test` - Run all tests with coverage
- `/test --unit` - Run only fast unit tests
- `/test tests/unit/llm_clients/` - Run specific test directory
- `/test --no-cov` - Run without coverage (faster)
