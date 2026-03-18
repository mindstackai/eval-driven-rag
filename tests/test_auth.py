"""
Tests for src/auth.py — role management logic.

All tests are isolated from the real roles.yaml by patching ROLES_PATH
to a temporary file via pytest's tmp_path fixture.
"""
import pytest
import yaml

import src.auth as auth
from src.auth import (
    VALID_ROLES,
    get_user_role,
    get_user_roles,
    list_users,
    remove_user,
    set_user_role,
)


@pytest.fixture(autouse=True)
def isolated_roles(tmp_path, monkeypatch):
    """Redirect all auth I/O to a temporary roles.yaml so tests don't
    touch the real file and don't interfere with each other."""
    fake_path = tmp_path / "roles.yaml"
    monkeypatch.setattr(auth, "ROLES_PATH", fake_path)
    yield fake_path


# ──────────────────────────────────────────────
# Default / empty state
# ──────────────────────────────────────────────

class TestDefaults:
    def test_unknown_user_defaults_to_public(self):
        assert get_user_role("nobody") == "public"

    def test_unknown_user_roles_defaults_to_public_list(self):
        assert get_user_roles("nobody") == ["public"]

    def test_list_users_empty_when_no_file(self):
        assert list_users() == {}


# ──────────────────────────────────────────────
# set_user_role / get_user_role
# ──────────────────────────────────────────────

class TestSetAndGet:
    def test_set_and_retrieve_role(self):
        set_user_role("alice", "admin")
        assert get_user_role("alice") == "admin"

    def test_overwrite_existing_role(self):
        set_user_role("bob", "analyst")
        set_user_role("bob", "public")
        assert get_user_role("bob") == "public"

    def test_multiple_users_isolated(self):
        set_user_role("alice", "admin")
        set_user_role("bob", "analyst")
        assert get_user_role("alice") == "admin"
        assert get_user_role("bob") == "analyst"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Invalid role"):
            set_user_role("alice", "superuser")

    def test_all_valid_roles_accepted(self):
        for role in VALID_ROLES:
            set_user_role("tester", role)
            assert get_user_role("tester") == role


# ──────────────────────────────────────────────
# Role hierarchy expansion
# ──────────────────────────────────────────────

class TestRoleHierarchy:
    def test_admin_can_read_all_roles(self):
        set_user_role("alice", "admin")
        roles = get_user_roles("alice")
        assert set(roles) == {"admin", "analyst", "public"}

    def test_analyst_cannot_read_admin(self):
        set_user_role("bob", "analyst")
        roles = get_user_roles("bob")
        assert "admin" not in roles
        assert set(roles) == {"analyst", "public"}

    def test_public_reads_only_public(self):
        set_user_role("carol", "public")
        roles = get_user_roles("carol")
        assert roles == ["public"]

    def test_hierarchy_order_admin_first(self):
        """Admin's expanded list should start with 'admin'."""
        set_user_role("alice", "admin")
        roles = get_user_roles("alice")
        assert roles[0] == "admin"


# ──────────────────────────────────────────────
# remove_user
# ──────────────────────────────────────────────

class TestRemoveUser:
    def test_removed_user_reverts_to_public(self):
        set_user_role("alice", "admin")
        remove_user("alice")
        assert get_user_role("alice") == "public"

    def test_remove_nonexistent_user_is_silent(self):
        remove_user("ghost")  # must not raise

    def test_remove_does_not_affect_other_users(self):
        set_user_role("alice", "admin")
        set_user_role("bob", "analyst")
        remove_user("alice")
        assert get_user_role("bob") == "analyst"


# ──────────────────────────────────────────────
# list_users
# ──────────────────────────────────────────────

class TestListUsers:
    def test_returns_all_users(self):
        set_user_role("alice", "admin")
        set_user_role("bob", "analyst")
        users = list_users()
        assert users == {"alice": "admin", "bob": "analyst"}

    def test_reflects_removal(self):
        set_user_role("alice", "admin")
        remove_user("alice")
        assert "alice" not in list_users()


# ──────────────────────────────────────────────
# Persistence — data survives across calls
# ──────────────────────────────────────────────

class TestPersistence:
    def test_role_written_to_yaml(self, isolated_roles):
        set_user_role("alice", "admin")
        with open(isolated_roles) as f:
            data = yaml.safe_load(f)
        assert data["users"]["alice"] == "admin"

    def test_hierarchy_preserved_on_write(self, isolated_roles):
        """set_user_role must not erase the hierarchy section."""
        set_user_role("alice", "admin")
        set_user_role("bob", "analyst")
        # hierarchy should still be readable from the file
        roles = get_user_roles("alice")
        assert "analyst" in roles
