Help create a new test file following VERA-MH testing conventions:

1. Ask user which module/file they want to test
2. Ask which test layer: unit, integration, or e2e
3. Analyze the target file:
   - Identify testable functions and classes
   - Detect async functions (need @pytest.mark.asyncio)
   - Identify LLM client usage (suggest MockLLM)
   - Detect file I/O operations (suggest tmp_path fixture)
4. Determine appropriate test directory:
   - Unit tests: `tests/unit/[module]/test_[filename].py`
   - Integration tests: `tests/integration/test_[feature].py`
   - E2E tests: `tests/e2e/test_[workflow].py`
5. Create test file with:
   - Proper imports (pytest, fixtures from conftest)
   - MockLLM import if needed
   - pytest.mark.asyncio for async tests
   - Appropriate fixtures (tmp_path, mock_llm, etc.)
   - Test stubs for each function with descriptive docstrings
6. Add test markers (@pytest.mark.unit, etc.)
7. Provide guidance on:
   - What to test (happy path, edge cases, errors)
   - How to use MockLLM for this project
   - How to handle async code
   - Relevant fixtures from conftest.py

Example patterns to include:
- MockLLM usage for LLM-dependent code
- tmp_path for file operations
- @pytest.mark.asyncio for async functions
- Proper test organization following test pyramid
