# Chageck

A system for automatically analyzing movie and TV series scripts for age ratings (0+, 6+, 12+, 16+, 18+) in accordance with Law of the Republic of Kazakhstan on the Protection of Children from Information Harmful to Their Health and Development.

Uses *LangChain* + *Ollama* for NLP, *Qdrant* for vector search, and *Celery* for asynchronous file processing.

---

## Functions:
- Upload PDF and DOCX scripts via the web interface.
- Asynchronous processing with progress indicator.
- Determine the age rating of a film.
- Classification by category:
    - Sex & Nudity
    - Violence & Gore
    - Profanity
    - Alcohol, Drugs & Smoking
    - Frightening & Intense Scenes

---

## Installation:
1. Launch VM
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Set dependencies
```
pip install -r requirements.txt
```

3. Configure `.env` 

4. Run services (redis, Qdrant)
```
docker compose up -d
```

5. Run script to add law info to Qdrant
```
python rag_law.py
```

6. Run backend
```
uvicorn app.main:app --reload
celery -A app.tasks.celery_app worker --loglevel=info
```

7. Run frontend
```
cd frontend
npm install
npm run dev
```

---

## Structure of project
```
.
├── app
│   ├── config.py
│   ├── jsonformer.py
│   ├── main.py
│   ├── pipeline.py
│   ├── rag_law.py
│   └── tasks.py
├── data
│   └── law.pdf
├── docker-compose.yml
├── frontend
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── public
│   │   └── vite.svg
│   ├── README.md
│   ├── src
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── assets
│   │   │   └── react.svg
│   │   ├── index.css
│   │   └── main.jsx
│   ├── tailwind.config.js
│   └── vite.config.js
├── prompts
│   └── classify_prompt.txt
├── README.md
└── requirements.txt
```
