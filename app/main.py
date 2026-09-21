import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models
from .database import engine, ensure_columns
from .routers import auth_router, signals_router, backtest_router, profile_router, market_router, alerts_router

logging.basicConfig(level=logging.INFO)

models.Base.metadata.create_all(bind=engine)
ensure_columns()  # adds new v2 columns to an existing goldsignal.db

app = FastAPI(title="GoldSignal API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend origin in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(signals_router.router)
app.include_router(alerts_router.router)
app.include_router(market_router.router)
app.include_router(backtest_router.router)
app.include_router(profile_router.router)


@app.get("/")
def root():
    from .data_fetcher import DEMO
    return {"status": "ok", "service": "GoldSignal API", "version": "2.0.0", "demo_data": DEMO}
