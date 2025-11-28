# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from quantum_engine import run_quantum_pricing  # this is your quantum code wrapper


app = FastAPI(
    title="ZyQ Labs – Quantum Option Pricing API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- CORS so the React app can call us from the browser ---
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request model ----

class PricingRequest(BaseModel):
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    N_steps: int = 50
    shots: int = 4096
    calibrate: bool = True


@app.get("/")
def root():
    return {"message": "ZyQ Labs Quantum Pricing API – use POST /price"}


@app.post("/price")
def price_endpoint(req: PricingRequest):
    """
    Run the quantum pricing engine and return:
      - inputs, model_params, outputs, errors, histogram
    exactly like in Swagger.
    """
    result = run_quantum_pricing(
    S0=req.S0,
    K=req.K,
    r=req.r,
    sigma=req.sigma,
    T=req.T,
    N_steps=req.N_steps,
    shots=req.shots,
    calibrate=req.calibrate,
)

    return result
