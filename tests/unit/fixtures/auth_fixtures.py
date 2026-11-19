"""
Authentication fixtures for unit tests.

Provides mock user and admin objects for testing authentication and authorization.
"""

import uuid

import pytest
from httpx import Headers
from sqlmodel import Session

from database.auth.models import User, UserRole
from security.utils import hash_password
from tests.unit.shared.auth_helpers import make_auth_headers

PASSWORD123_HASH = hash_password("password123")
ADMINPASS123_HASH = hash_password("adminpass123")


@pytest.fixture
def mock_user(session: Session):
    """Create a mock developer user for testing."""
    user = User(
        id=uuid.uuid4(),
        username="testuser",
        password=PASSWORD123_HASH,
        role=UserRole.DEVELOPER,
        enabled=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@pytest.fixture
def mock_user_disabled(session: Session):
    """Create a mock developer user for testing."""
    user = User(
        id=uuid.uuid4(),
        username="testuser_disabled",
        password=PASSWORD123_HASH,
        role=UserRole.DEVELOPER,
        enabled=False,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@pytest.fixture
def mock_admin(session: Session):
    """Create a mock admin user for testing."""
    admin = User(
        id=uuid.uuid4(),
        username="admin",
        password=ADMINPASS123_HASH,
        role=UserRole.ADMINISTRATOR,
        enabled=True,
    )
    session.add(admin)
    session.commit()
    session.refresh(admin)
    return admin


@pytest.fixture
def admin_headers(session: Session, mock_admin: User) -> Headers:
    """Authorization headers of a signed-in administrator."""
    return make_auth_headers(mock_admin, session)


@pytest.fixture
def user_headers(session: Session, mock_user: User) -> Headers:
    """Authorization headers of a signed-in developer."""
    return make_auth_headers(mock_user, session)
