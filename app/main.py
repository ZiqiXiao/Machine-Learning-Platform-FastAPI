from fastapi import FastAPI

from config.settings import app_logger as logger

app = FastAPI()


@app.get("/hello-world")
def hello_world() -> dict[str, str]:
    logger.info("hello world")
    return {"message": "Hello World!"}
