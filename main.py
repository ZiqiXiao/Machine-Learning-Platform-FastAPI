from fastapi import FastAPI

from app.routers import ml
from config.settings import app_logger as logger

app = FastAPI()
app.include_router(ml.router)


@app.get("/hello-world")
async def hello_world() -> dict[str, str]:
    logger.info("hello world")
    return {"message": "Hello World!"}
