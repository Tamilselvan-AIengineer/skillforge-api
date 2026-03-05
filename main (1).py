from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="SkillForge AI API",
    description="RAG-powered Career Mentor API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────
class MentorRequest(BaseModel):
    query: str
    student_skills: List[str]
    goal: str

class MentorResponse(BaseModel):
    answer: str
    retrieved_context: str
    missing_skills: List[str]

class UpdateSkillRequest(BaseModel):
    student_skills: List[str]
    new_skill: str
    goal: str

class SkillGapRequest(BaseModel):
    student_skills: List[str]
    goal: str

class SkillGapResponse(BaseModel):
    missing_skills: List[str]
    readiness_percent: int
    priority_order: List[str]

# ── Knowledge Base ────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "Data Scientist": {
        "required": [
            "Python", "Statistics", "Machine Learning",
            "Deep Learning", "SQL", "Data Visualization",
            "Feature Engineering", "Model Deployment"
        ],
        "docs": [
            """Data Scientist requires: Python, Statistics, Machine Learning,
            Deep Learning, SQL, Data Visualization, Feature Engineering.
            Key projects: Customer Churn Prediction, NLP Sentiment Analysis,
            Real-time Dashboard, ML Model Deployment Pipeline.
            Interview topics: bias-variance tradeoff, regularization,
            cross-validation, feature importance, precision vs recall.""",
            """Learning roadmap for Data Scientist:
            Month 1-2: Python, Pandas, NumPy, SQL, Basic Statistics.
            Month 3-4: Scikit-learn, Supervised Learning, Model Evaluation.
            Month 5-6: Deep Learning, PyTorch, HuggingFace Transformers.
            Month 7-8: Model Deployment, Docker, FastAPI, Cloud platforms.
            Month 9+: Kaggle competitions, portfolio, job applications.""",
        ]
    },
    "ML Engineer": {
        "required": [
            "Python", "Machine Learning", "Deep Learning", "MLOps",
            "Docker", "Kubernetes", "Data Pipelines",
            "Cloud Platforms", "REST APIs", "System Design"
        ],
        "docs": [
            """Machine Learning Engineer requires: Python, MLOps, Docker,
            Kubernetes, Data Pipelines, Cloud Platforms, REST APIs.
            Key tools: MLflow, DVC, FastAPI, AWS SageMaker, Kubeflow.
            Interview topics: model drift, CI/CD for ML, online vs batch
            inference, feature stores, distributed training.""",
            """ML Engineer roadmap:
            Month 1-3: Advanced ML, PyTorch, model optimization.
            Month 4-6: MLOps with MLflow, DVC, Docker, FastAPI.
            Month 7-9: Cloud (AWS/GCP), Kubernetes, Kubeflow.
            Month 10+: Feature stores, model monitoring, A/B testing.""",
        ]
    },
    "Software Engineer": {
        "required": [
            "Data Structures", "Algorithms", "System Design",
            "Java/Python/C++", "REST APIs", "Databases", "Git", "Testing"
        ],
        "docs": [
            """Software Engineer requires: Data Structures, Algorithms,
            System Design, REST APIs, Databases, Git, Testing, CI/CD.
            Key projects: URL Shortener, Chat App, Task Queue, Code Review Bot.
            Interview topics: linked lists, LRU cache, CAP theorem,
            SOLID principles, system design patterns.""",
            """Software Engineer roadmap:
            Month 1-2: DSA fundamentals, Big-O, sorting/searching.
            Month 3-4: OOP, design patterns, databases, REST APIs.
            Month 5-6: System design, microservices, Docker.
            Month 7+: LeetCode 150, mock interviews, portfolio.""",
        ]
    }
}

GENERAL_DOCS = [
    """Placement preparation tips:
    Resume: Include 3 strong projects with GitHub links and metrics.
    LinkedIn: Optimize headline, add skills, get recommendations.
    GitHub: Pin best repos, write clear READMEs, consistent commits.
    Interviews: Practice STAR method for behavioral questions.
    Networking: Connect with 5 professionals per week on LinkedIn.""",
    """Higher studies options:
    MS Data Science: CMU, Stanford, University of Michigan, NYU.
    MS Computer Science: MIT, UC Berkeley, Georgia Tech (online).
    MS Statistics: Columbia, Duke, University of Chicago.
    GRE required for most programs. Strong GPA + projects + LORs needed.""",
    """Free learning resources:
    Python/ML: Kaggle Learn, fast.ai, Andrew Ng Coursera (audit free).
    DSA: LeetCode free tier, HackerRank, NeetCode YouTube.
    System Design: System Design Primer GitHub, ByteByteGo YouTube.
    Projects: Build and deploy on GitHub, write case studies on Medium.""",
]

# ── Initialize Embeddings + Vector Store ─────────────────────
print("⏳ Loading embedding model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
all_docs = []

for role, data in KNOWLEDGE_BASE.items():
    for doc in data["docs"]:
        chunks = splitter.create_documents([doc], metadatas=[{"role": role}])
        all_docs.extend(chunks)

for doc in GENERAL_DOCS:
    chunks = splitter.create_documents([doc], metadatas=[{"role": "general"}])
    all_docs.extend(chunks)

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings_model,
    persist_directory="./skillforge_db"
)
vectorstore.persist()
print(f"✅ Vector store ready with {len(all_docs)} chunks!")

# ── Groq Client ───────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("✅ Groq client ready!")

# ── Helpers ───────────────────────────────────────────────────
def retrieve_context(query: str, student_skills: list, top_k: int = 3) -> str:
    enriched = f"{query} skills: {', '.join(student_skills)}"
    docs = vectorstore.similarity_search(enriched, k=top_k)
    return "\n\n".join([d.page_content for d in docs])

def get_missing_skills(student_skills: list, goal: str) -> list:
    if goal not in KNOWLEDGE_BASE:
        return []
    required = KNOWLEDGE_BASE[goal]["required"]
    lower_skills = [s.lower() for s in student_skills]
    return [r for r in required if r.lower() not in lower_skills]

def generate_answer(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI Career Mentor for students "
                    "targeting tech jobs. Give structured, actionable "
                    "advice using bullet points. Be encouraging and specific."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.7
    )
    return response.choices[0].message.content

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "🎯 SkillForge AI API is running!",
        "version": "1.0.0",
        "endpoints": ["/ask", "/skill-gap", "/update-skill", "/projects/{goal}", "/roadmap/{goal}", "/interview-prep/{goal}"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "vectorstore": "ready", "llm": "groq"}

@app.post("/ask", response_model=MentorResponse)
async def ask_mentor(request: MentorRequest):
    try:
        context = retrieve_context(request.query, request.student_skills)
        missing = get_missing_skills(request.student_skills, request.goal)
        prompt = f"""
Student Profile:
- Current Skills: {', '.join(request.student_skills)}
- Target Role: {request.goal}
- Missing Skills: {', '.join(missing) if missing else 'None!'}

Retrieved Knowledge (RAG):
{context}

Student Question: {request.query}

Give personalized advice referencing their specific skill gaps.
Use bullet points. Keep response under 300 words.
"""
        answer = generate_answer(prompt)
        return MentorResponse(answer=answer, retrieved_context=context, missing_skills=missing)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skill-gap", response_model=SkillGapResponse)
async def get_skill_gap(request: SkillGapRequest):
    try:
        missing = get_missing_skills(request.student_skills, request.goal)
        if request.goal in KNOWLEDGE_BASE:
            required = KNOWLEDGE_BASE[request.goal]["required"]
            have = len(required) - len(missing)
            readiness = round((have / len(required)) * 100)
        else:
            readiness = 0
        return SkillGapResponse(missing_skills=missing, readiness_percent=readiness, priority_order=missing)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-skill")
async def update_skill(request: UpdateSkillRequest):
    try:
        new_doc = Document(
            page_content=f"Student has learned: {request.new_skill}. This is now part of their skillset for {request.goal}.",
            metadata={"type": "continual_learning", "skill": request.new_skill}
        )
        vectorstore.add_documents([new_doc])
        vectorstore.persist()
        updated_skills = request.student_skills + [request.new_skill]
        missing = get_missing_skills(updated_skills, request.goal)
        prompt = f"""
Student just learned: {request.new_skill}
Updated Skills: {', '.join(updated_skills)}
Goal: {request.goal}
Still Missing: {', '.join(missing) if missing else 'Nothing — fully skilled!'}

In 3 bullet points tell them:
1. What to learn next
2. A project to build using {request.new_skill}
3. How close they are to job-ready
"""
        next_steps = generate_answer(prompt)
        return {"message": f"✅ '{request.new_skill}' added!", "updated_skills": updated_skills, "missing_skills": missing, "next_steps": next_steps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{goal}")
async def get_projects(goal: str, student_skills: str = ""):
    try:
        skills_list = student_skills.split(",") if student_skills else []
        missing = get_missing_skills(skills_list, goal)
        prompt = f"""
Goal: {goal}
Student Skills: {student_skills or 'beginner'}
Missing Skills: {', '.join(missing)}
Suggest exactly 3 projects. For each: name, skills taught, difficulty, duration, 2-sentence description.
Format as numbered list.
"""
        projects = generate_answer(prompt)
        return {"goal": goal, "projects": projects, "missing_skills": missing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/roadmap/{goal}")
async def get_roadmap(goal: str, student_skills: str = ""):
    try:
        skills_list = student_skills.split(",") if student_skills else []
        missing = get_missing_skills(skills_list, goal)
        context = retrieve_context(f"learning roadmap for {goal}", skills_list)
        prompt = f"""
Goal: {goal}
Current Skills: {student_skills or 'beginner'}
Missing Skills: {', '.join(missing)}
Context: {context}
Create a month-by-month roadmap (6 months).
Each month: what to learn, free resources, mini project.
"""
        roadmap = generate_answer(prompt)
        return {"goal": goal, "roadmap": roadmap, "missing_skills": missing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interview-prep/{goal}")
async def interview_prep(goal: str):
    try:
        context = retrieve_context(f"interview preparation {goal}", [])
        prompt = f"""
Goal: {goal}
Context: {context}
Give interview preparation:
- 5 technical questions with brief answers
- 3 behavioral questions with STAR tips
- Resume advice
- Day-before checklist
"""
        tips = generate_answer(prompt)
        return {"goal": goal, "interview_prep": tips}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
