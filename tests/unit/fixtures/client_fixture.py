import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlmodel import Session

import database.engine
from database.general import get_session
from main import app


@pytest.fixture(scope="session", name="client")
def client(engine: Engine):
    def _get_session():
        yield Session(engine)

    app.dependency_overrides[get_session] = _get_session
    original_engine = database.engine.general_engine
    database.engine.general_engine = engine

    yield TestClient(app)
    database.engine.general_engine = original_engine
    app.dependency_overrides.clear()
