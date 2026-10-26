# Agentic STEM Tutoring

Agentic STEM Tutoring is an intelligent tutoring prototype for programming-oriented STEM education.

The system combines a FastAPI backend, React/Vite frontend, SQLite database, SQLAlchemy ORM, Qwen/DashScope LLM integration, fuzzy learner-state estimation, explicit pedagogical planning, and experimental evaluation.

## Project Structure

```text
stem-editor/
├── backend/
├── frontend/
└── README.md
```

Main components:

- `backend/` — FastAPI backend, tutoring logic, learner modeling, and experiments
- `frontend/` — React/Vite frontend
- `backend/experiment_data/` — controlled benchmark definitions
- `backend/experiment_results/` — archived experimental outputs
- `backend/scripts/` — experiment and utility scripts
- `backend/pedagogy/` — belief-state estimation, planning, reward, transition, policy, and strategy logic
- `backend/fuzzy/` — fuzzy confusion estimation

## Requirements

Recommended environment:

```text
Python 3.11+
Node.js 18+
npm 9+
```

Backend dependencies are listed in:

```text
backend/requirements.txt
```

Frontend dependencies are listed in:

```text
frontend/package.json
```

## Backend Setup

Create a Python virtual environment from the project root.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Create:

```text
backend/.env
```

Example:

```env
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=YOUR_API_KEY
QWEN_MODEL=qwen3.7-plus
```

Replace `YOUR_API_KEY` with your own API credential.

Do not commit a real API key to a public repository.

Start the backend from the `backend/` directory:

```bash
uvicorn main:app --reload
```

If `uvicorn` is not found:

```bash
python -m uvicorn main:app --reload
```

Backend address:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

Open another terminal and run:

```bash
cd frontend
npm install
npm run dev
```

Frontend address:

```text
http://localhost:5173
```

## Experimental Materials

The repository includes materials supporting the experiments reported in the associated manuscript.

Controlled benchmark definitions:

```text
backend/experiment_data/
```

Archived experimental outputs:

```text
backend/experiment_results/
```

Experiment scripts:

```text
backend/scripts/
```

The archived materials include results for:

- controlled baseline comparison
- learner-state sensitivity analysis
- teacher hint-constraint stress testing
- parameter sensitivity analysis
- context-matched pilot experiments
- automated response screening

The archived CSV files can be inspected independently of the interactive tutoring application.

Some experiments involving external LLM calls require a valid API credential and may produce different generated text, latency, or token usage when rerun at a later time.

Deterministic components such as pedagogical planning and parameter analysis can be inspected directly from the implementation and archived outputs.

## Database

The prototype uses SQLite with SQLAlchemy.

Because the default database path is relative to the backend working directory, start the backend from:

```text
backend/
```

Database files are local runtime data and are not required for inspecting the archived experimental CSV outputs.

## Security

Do not upload private credentials or sensitive local data to a public repository.

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
node_modules/
.DS_Store
*.db
```

In particular, never publish a real:

```text
DASHSCOPE_API_KEY
```

## Platform Note

Python virtual environments are platform-specific.

Do not copy the same `venv/` between Windows and macOS.

Instead, create a new virtual environment on each machine and reinstall dependencies:

```bash
pip install -r requirements.txt
```

## Development Addresses

| Service | Address |
|---|---|
| Frontend | http://localhost:5173 |
| Backend | http://127.0.0.1:8000 |
| Swagger | http://127.0.0.1:8000/docs |

## Research Reproducibility

This repository contains implementation code, benchmark definitions, experiment scripts, and archived result tables used to support the manuscript analyses.

The archived experimental results are provided for inspection and reproducibility. External LLM-dependent experiments may vary when rerun because model outputs and service latency can change over time.

## License

This repository is provided for research and review purposes.