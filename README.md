# VERA-MH

[![CI](https://github.com/SpringCare/VERA-MH/workflows/CI/badge.svg)](https://github.com/SpringCare/VERA-MH/actions/workflows/ci.yml)
[![Docker](https://github.com/SpringCare/VERA-MH/workflows/Docker%20Build%20Validation/badge.svg)](https://github.com/SpringCare/VERA-MH/actions/workflows/docker.yml)

VERA-MH (Validation of Ethical and Responsible AI in Mental Health) is a comprehensive framework for evaluating AI systems designed for mental health applications. This toolkit enables researchers, developers, and clinicians to systematically assess how well AI systems handle sensitive mental health conversations across detecting potential risk, confirming risk, guiding to human support, communicaticating effectively, and holding safe boundaries. By simulating realistic patient-provider interactions using clinically-developed personas and rubrics, VERA-MH provides standardized evaluation metrics that help ensure AI mental health tools are safe, effective, and responsible before deployment.

This code should be considered a work in progress (including this documentation), and the main avenue to offer feedback.
We value every interaction that follows the [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
There are many quirks of the current structure, which will be simplified and streamlined.

## Table of Contents

- [Getting Started](#getting-started)
- [Using Extra Parameters](#using-extra-parameters)
- [Data Files](#data-files)
- [LLM Conversation Simulator](#llm-conversation-simulator)
- [Using Claude Code](#using-claude-code)
- [License](#license)

## Additional Resources

<!-- Placeholder: Link to blog post or VERA-MH website -->
<!-- Placeholder: Link to Kate's paper (publication details TBD) -->
<!-- Placeholder: AI version of the paper - NeurIPS -->
<!-- Placeholder: Other content from Kelly -->

# Getting started
0. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

1. **Set up environment and install dependencies**:
   ```bash
   uv sync
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys (e.g., OpenAI/Anthropic)
   ```

3. **(Optional) Install pre-commit hooks** for automatic code formatting/linting:
   ```bash
   pre-commit install
   ```

4. **(Optional) Create an LLM class for your agent**: see guidance [here](docs/evaluating.MD)

5. **Run the simulation** (quick test with 6 turns for cost-effective trial):
   ```bash
   python generate.py -u gpt-4o -uep temperature=1 -p gpt-4o -pep temperature=1 -t 6 -r 1
   ```
   
   **5a. Quick test**: The command above generates a small set of conversations for initial testing.
   
   **5b. For production-quality evaluations**: To generate conversations that reproduce published VERA scores, achieve valid scoring, or use scoring features, we recommend:
   ```bash
   python generate.py -u gemini-3-pro-preview -p <your-AI-product> -pep <your-AI-product-extras> -t 20 -r 20 -c 10
   ```
   - **20 conversation turns** over **20 runs** per persona for reliable scoring
   - **Max concurrent** of **10** conversations (use `-c 10`) to manage API rate limits
   - **Model recommendation**: **Gemini 3 Pro** makes the most realistic conversations as evaluated by our clinicians

**Parameters for `generate.py`:**

| Short | Full | Description |
|-------|------|-------------|
| `-u` | `--user-agent` | Model for the user-agent (persona). Examples: `claude-3-5-sonnet-20241022`, `gemini-1.5-pro`, `llama3:8b` |
| `-uep` | `--user-agent-extra-params` | Extra parameters for the user-agent. Examples: `temperature=0.7`, `max_tokens=1000` |
| `-p` | `--provider-agent` | Model for the provider-agent (AI system being evaluated). Examples: `claude-3-5-sonnet-20241022`, `gemini-1.5-pro`, `llama3:8b` |
| `-pep` | `--provider-agent-extra-params` | Extra parameters for the provider-agent. Examples: `temperature=0.7`, `max_tokens=1000` |
| `-t` | `--turns` | Number of turns per conversation (required) (e.g., 2 turns means both persona and provider spoke once) |
| `-r` | `--runs` | Number of runs per prompt (required) |
| `-f` | `--folder-name` | Folder name for output (defaults to `conversations` with a subfolder named based on other parameters and datetime) |
| `-c` | `--max-concurrent` | Maximum number of concurrent conversations (defaults to None; use this if the provider you're testing times out) |
| `-w` | `--max-total-words` | Optional maximum total words across all responses in a conversation |
| `-i` | `--run-id` | Run ID for the conversations (if not provided, a default will be generated) |
| `-mp` | `--max-personas` | Maximum number of personas to use (limits personas loaded from [data/personas.tsv](data/personas.tsv)) |
| `-d` | `--debug` | Enable debug logging for conversation generation |

This will generate conversations and store them in a subfolder of `conversations` unless specified otherwise.

6. **Judge the conversations**:
   ```bash
   python judge.py -f conversations/{YOUR_FOLDER} -j gpt-4o
   ```

**Judge model recommendations**: **GPT-4o** and **Claude Sonnet** have the highest inter-rater reliability with human clinicians as judge models.

**Parameters for `judge.py`:**

| Short | Full | Description |
|-------|------|-------------|
| `-f` | `--folder` | Folder containing conversation files (default: `conversations`) |
| `-c` | `--conversation` | Path to a single conversation file to judge (mutually exclusive with `--folder`) |
| `-j` | `--judge-model` | Model(s) to use for judging (required). Format: `model` or `model:count` for multiple instances. Can specify multiple: `--judge-model model1 model2:3`. Examples: `claude-3-5-sonnet-20241022`, `claude-3-5-sonnet-20241022:3`, `claude-3-5-sonnet-20241022:2 gpt-4o:1` |
| `-jep` | `--judge-model-extra-params` | Extra parameters for the judge model (optional). Examples: `temperature=0.7`, `max_tokens=1000`. Default: `temperature=0` (unless overridden) |
| `-r` | `--rubrics` | Rubric file(s) to use (default: `data/rubric.tsv`) |
| `-l` | `--limit` | Limit number of conversations to judge (for debugging) |
| `-o` | `--output` | Output folder for evaluation results (default: `evaluations`) |
| `-m` | `--max-concurrent` | Maximum number of concurrent workers (default: None). Set to a high number or omit for unlimited concurrency |
| `--per-judge` | | If set, `--max-concurrent` applies per judge model. Otherwise, it applies to total workers across all judges |
| `--verbose-workers` | | Enable verbose worker logging to show concurrency behavior |


### Using Extra Parameters

Both `generate.py` and `judge.py` support extra parameters for fine-tuning model behavior:

**Generate with temperature control:**
```bash
# Lower temperature (0.3) for more consistent responses
python generate.py -u gpt-4o -uep temperature=0.3 -p claude-3-5-sonnet-20241022 -pep temperature=0.5 -t 6 -r 2

# Higher temperature (1.0) with max tokens
python generate.py -u gpt-4o -uep temperature=1,max_tokens=2000 -p gpt-4o -pep temperature=1 -t 6 -r 1
```

**Judge with custom parameters:**
```bash
# Use lower temperature for more consistent evaluation
python judge.py -f conversations/my_experiment -j claude-3-5-sonnet-20241022 -jep temperature=0.3

# Multiple parameters
python judge.py -f conversations/my_experiment -j gpt-4o -jep temperature=0.5,max_tokens=1500
```

**Note:** Extra parameters are automatically included in the output folder names, making it easy to track experiments:
- Generation: `conversations/p_gpt_4o_temp0.3__a_claude_3_5_sonnet_temp0.5__t6__r2__{timestamp}/`
- Evaluation: `evaluations/j_claude_3_5_sonnet_temp0.3_{timestamp}__{conversation_folder}/`

**Multiple judge models**: You can use multiple different judge models and/or multiple instances:
```bash
# Multiple different models
python judge.py -f conversations/{YOUR_FOLDER} -j gpt-4o claude-sonnet-4-20250514

# Multiple instances of the same model (for reliability testing)
python judge.py -f conversations/{YOUR_FOLDER} -j gpt-4o:3

# Combine both: different models with multiple instances
python judge.py -f conversations/{YOUR_FOLDER} -j gpt-4o:2 claude-sonnet-4-20250514:3
```

## Data Files

Most of the interesting data is contained in the [`data`](data) folder, specifically:
- _personas.csv_ has the data for the personas
- *personas_prompt_template.txt* has the meta-prompt for the user-agent
- _rubric.csv_ is the clinically developed rubric
- *rubric_prompt_template.txt* for the judge meta prompt 


# LLM Conversation Simulator

VERA-MH simulates realistic conversations between Large Language Models (LLMs) for mental health care evaluation. The system uses clinically-developed personas and rubrics to generate patient-provider interactions, enabling systematic assessment of AI mental health tools. Conversations are generated between persona models (representing patients) and provider models (representing therapists), then evaluated by judge models against clinical rubrics to assess performance across multiple dimensions including risk detection, resource provision, and ethical considerations.

## Features

### Conversation Generation
- **Mental Health Personas**: CSV-based system with realistic patient personas including age, background, mental health context, and risk factors
- **Asynchronous Generation**: Concurrent conversation generation for efficient batch processing
- **Modular Architecture**: Abstract LLM interface allows for easy integration of different LLM providers
- **System Prompts**: Each LLM instance can be initialized with custom system prompts loaded from files
- **Early Stopping**: Conversations can end naturally when personas signal completion
- **Conversation Tracking**: Full conversation history is maintained with comprehensive logging
- **Batch Processing**: Run multiple conversations with different personas and multiple runs per persona

### Conversation Evaluation
- **LLM-based Judging**: Automated evaluation of conversations using LLM judges against clinical rubrics
- **Structured Output**: Uses Pydantic models and LangChain's structured output for reliable, type-safe responses
- **Question Flow Navigation**: Dynamic rubric navigation based on answers (with GOTO logic, END conditions, etc.)
- **Dimension Scoring**: Evaluates conversations across multiple clinical dimensions (risk detection, resource provision, etc.)
- **Severity Assessment**: Assigns severity levels (High/Medium/Low) based on rubric criteria
- **Comprehensive Logging**: Detailed logs of all judge decisions and reasoning

### LLM Provider Support
- **LangChain Integration**: Uses LangChain for robust LLM interactions
- **Claude Support**: Full implementation of Claude models via Anthropic's API with structured output
- **OpenAI Support**: Complete integration with GPT models via OpenAI's API with structured output
- **Gemini Support**: Google Gemini integration with structured output
- **Llama Support**: Local Llama models via Ollama (limited structured output support)


## Architecture

### Core Components

- **`generate.py`**: Main entry point for conversation generation with configurable parameters
- **`judge.py`**: Main entry point for evaluating conversations using LLM judges
- **`generate_conversations/`**: Core conversation generation system
  - **`conversation_simulator.py`**: Manages individual conversations between persona and agent LLMs
  - **`runner.py`**: Orchestrates multiple conversations with logging and file management
  - **`utils.py`**: CSV-based persona loading and prompt templating
- **`judge/`**: Conversation evaluation system
  - **`llm_judge.py`**: LLM-based judge for evaluating conversations against rubrics
  - **`response_models.py`**: Pydantic models for structured LLM responses
  - **`question_navigator.py`**: Navigates through rubric questions based on answers
  - **`score.py`**: Scoring logic for dimension evaluation
  - **`runner.py`**: Orchestrates judging of multiple conversations
  - **`utils.py`**: Utility functions for rubric loading and processing
- **`llm_clients/`**: LLM provider implementations with structured output support
  - **`llm_interface.py`**: Abstract base class defining the LLM interface
  - **`llm_factory.py`**: Factory class for creating LLM instances
  - **`claude_llm.py`**: Claude implementation using LangChain with structured output
  - **`openai_llm.py`**: OpenAI implementation with structured output
  - **`gemini_llm.py`**: Google Gemini implementation with structured output
  - **`llama_llm.py`**: Llama implementation via Ollama
  - **`config.py`**: Configuration management for API keys and model settings
- **`utils/`**: Utility functions and helpers
  - **`prompt_loader.py`**: Functions for loading prompt configurations
  - **`model_config_loader.py`**: Model configuration management
  - **`conversation_utils.py`**: Conversation formatting and file operations
  - **`logging_utils.py`**: Comprehensive logging for conversations
- **`data/`**: Persona and configuration data
  - **`personas.csv`**: CSV file containing patient persona data
  - **`persona_prompt_template.txt`**: Template for generating persona prompts
  - **`rubric.tsv`**: Clinical rubric for conversation evaluation
  - **`rubric_prompt_beginning.txt`**: System prompt for the judge
  - **`question_prompt.txt`**: Prompt template for asking rubric questions
  - **`model_config.json`**: Model assignments for different prompt types

### Persona System

The system uses a CSV-based approach for managing mental health patient personas:

#### Persona Data Structure (`data/personas.csv`)
Each persona includes:
- **Demographics**: Name, Age, Gender, Background
- **Mental Health Context**: Current mental health situation
- **Risk Assessment**: Risk Type (e.g., Suicidal Intent, Self Harm) and Acuity (Low/Moderate/High)
- **Communication Style**: How the persona expresses themselves
- **Triggers/Stressors**: What causes distress
- **Sample Prompt**: Example of what they might say

#### Prompt Templating (`data/persona_prompt_template.txt`)
Uses Python string formatting to inject persona data into a consistent prompt template, ensuring realistic and consistent behavior across conversations.

### Structured Output System

The judge evaluation system uses **structured output** to ensure reliable and type-safe responses from LLMs:

#### How It Works
1. **Pydantic Models** ([judge/response_models.py](judge/response_models.py)): Define the structure of expected responses
   ```python
   class QuestionResponse(BaseModel):
       answer: str  # The selected answer from valid options
       reasoning: str  # Explanation for the choice
   ```

2. **LLM Interface** ([llm_clients/llm_interface.py](llm_clients/llm_interface.py)): Abstract method for structured responses
   ```python
   async def generate_structured_response(
       self, message: str, response_model: Type[T]
   ) -> T:
       """Returns a Pydantic model instance instead of raw text"""
   ```

3. **Provider Implementation**: Each LLM client implements structured output using LangChain's `with_structured_output()`
   - Claude, OpenAI, and Gemini: Native structured output support via API
   - Llama: Limited support (may require prompt-based parsing)

#### Benefits
- ✅ **Type Safety**: Automatic validation of LLM responses
- ✅ **Reliability**: No fragile string parsing (`"ANSWER: ..."` → direct field access)
- ✅ **Consistency**: All providers return the same structured format
- ✅ **Error Handling**: Clear validation errors when LLM responses don't match schema

#### Usage in Judge
The judge uses structured output when asking rubric questions:
```python
# Instead of parsing "ANSWER: Yes\nREASONING: ..."
structured_response = await evaluator.generate_structured_response(
    prompt, QuestionResponse
)
answer = structured_response.answer  # Direct access
reasoning = structured_response.reasoning  # Type-safe
```

## Usage

### Basic Conversation Generation

```python
from generate import generate_conversations

# Persona model configuration (the "patient")
persona_model_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Agent model configuration (the "therapist")
agent_model_config = {
    "model": "claude-sonnet-4-20250514",
    "prompt_name": "therapist",  # Must match a prompt config file
    "name": "Claude Sonnet",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Generate conversations
results = await generate_conversations(
    persona_model_config=persona_model_config,
    agent_model_config=agent_model_config,
    max_turns=5,
    runs_per_prompt=3,
    persona_names=["Alex M.", "Chloe Kim"],  # Optional: filter specific personas
    folder_name="custom_experiment"  # Optional: custom output folder
)
```

### Command Line Usage

```bash
python generate.py
```

The script will:
1. Load personas from `data/personas.csv`
2. Generate conversations between each persona and the agent
3. Run multiple iterations per persona (configurable)
4. Save conversations and logs to timestamped folders
5. Support early termination when personas indicate completion


### Custom Personas and Prompts

#### 1. Add New Personas (`data/personas.csv`)
Add new rows to the CSV file with the required fields:

```csv
Name,Age,Gender,Background,Mental Health Context,Communication Style,Trajectory of sharing,Sample Prompt,Triggers/Stressors,Risk Type,Acuity
New Patient,30,Female,Software engineer,Experiencing burnout,Direct and analytical,Open about work stress,"I can't focus at work anymore",Work pressure deadlines,Self Harm,Moderate Acuity
```

#### 2. Modify Prompt Template (`data/persona_prompt_template.txt`)
Update the template to include new fields or modify behavior patterns.

#### 3. Configure Models (`model_config.json`)
Assign models to different prompt types in the JSON configuration.

### Early Stopping

The conversation simulator supports natural conversation termination when personas signal completion:

**Termination Signals Detected:**
- Explicit endings: "Thank you, I'm done", "goodbye", "bye", "farewell"
- Natural conclusions: "in conclusion", "to conclude", "final thoughts"
- Polite endings: "thanks for", "pleasure talking", "great conversation"
- Direct signals: "i'm done", "let's end here", "nothing more to discuss"

**How It Works:**
1. Only personas (conversation initiators) can trigger early termination
2. Conversations require at least 3 turns before termination is allowed
3. When termination signals are detected, the conversation ends immediately
4. Both console output and saved files indicate early termination

## Configuration

### Model Settings

Model settings can be adjusted in the configuration dictionaries:

```python
persona_model_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.7,        # Controls response creativity
    "max_tokens": 1000         # Maximum response length
}
```

### Output Organization

Conversations are organized into timestamped folders by default:

```
conversations/
├── p_claude_sonnet_4_20250514__a_claude_sonnet_4_20250514_20250120_143022_t5_r3/
│   ├── abc123_Alex_M_c3s_run1_20250120_143022_123.txt
│   ├── def456_Chloe_Kim_c3s_run1_20250120_143022_456.txt
```

### Logging

Comprehensive logging tracks:
- Conversation start/end times
- Each turn with speaker, input, and response
- Early termination events
- Performance metrics (duration, turn count)
- Error handling and debugging information

Conversation logs are organized into timestamped folders by default:

```
logging/
├── p_claude_sonnet_4_20250514__a_claude_sonnet_4_20250514_t5_r3_20250120_143022/
│   ├── abc123_Alex_M_c3s_run1.log
│   └── def456_Chloe_Kim_c3s_run1.log
```

# Development with Agents
This project has multiple instructions/insights for agents to utilize as helpful context in assisting development.
See [AGENTS.md](./AGENTS.md) for more info.


## Claude Development
This project is configured with [Claude Code](https://claude.ai/claude-code), Anthropic's CLI tool that helps with development tasks.

If you have Claude Code installed, you can use these custom commands:

**Development & Setup:**
- `/setup-dev` - Set up complete development environment (includes test infrastructure)

**Code Quality:**
- `/format` - Run code formatting and linting (ruff + pyright)

**Running VERA-MH:**
- `/run-generator` - Interactive conversation generator
- `/run-judge` - Interactive conversation evaluator

**Testing:**
- `/test` - Run test suite (with coverage by default)
- `/fix-tests` - Fix failing tests iteratively, show branch-focused coverage
- `/create-tests [module_path] [--layer=unit|integration|e2e]` - Create tests (focused: single module, or coverage analysis: find and fix gaps)

**Git Workflow:**
- `/create-commits` - Create logical, organized commits (with optional branch creation)
- `/create-pr` - Create GitHub pull request with auto-generated summary

## Configuration
See [CLAUDE.md](./CLAUDE.md) and [.claude/](./.claude/) for all options.

Team-shared configuration is in [`.claude/settings.json`](./.claude/settings.json), which defines allowed operations without approval. Personal settings can be added to `.claude/settings.local.json` (not committed to git).

For more details on custom commands and creating your own, see [`.claude/commands/README.md`](.claude/commands/README.md).

# License

We use a MIT license with conditions. We changed the reference from "software" to "materials" and more accurately describe the nature of the project. See the [LICENSE](./LICENSE) file for full details.
