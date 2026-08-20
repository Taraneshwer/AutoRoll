# AutoRoll Development Guide

## Local Setup Instructions

1. **Clone & Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Environment Configuration**:
   ```bash
   cp .env.example .env
   ```

4. **Running Central Server**:
   ```bash
   python -m server.main
   ```

5. **Running Standalone Worker**:
   ```bash
   python -m worker.main
   ```

6. **Running Unit Tests**:
   ```bash
   pytest
   ```
