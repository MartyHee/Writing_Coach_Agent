"""Development launcher for the installed Writing Coach Agent package."""
from writing_coach_agent.main import app, settings


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
