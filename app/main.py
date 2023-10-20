from fastapi import FastAPI

app = FastAPI()


@app.get("/hello-world")
def hello_world() -> dict[str, str]:
    return {"message": "Hello World!"}
