"""
auth.py — Provider-agnostic access control.

Roles are stored in roles.yaml at the project root.
No cloud dependencies — swap the backend by replacing _load_roles / _save_roles.

Role hierarchy (default):
  admin   → can read admin, analyst, and public chunks
  analyst → can read analyst and public chunks
  public  → can read public chunks only
"""
from pathlib import Path
import yaml

ROLES_PATH = Path("roles.yaml")

VALID_ROLES = ["public", "analyst", "admin"]

_DEFAULT_HIERARCHY = {
    "admin":   ["admin", "analyst", "public"],
    "analyst": ["analyst", "public"],
    "public":  ["public"],
}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_data() -> dict:
    if not ROLES_PATH.exists():
        return {}
    with open(ROLES_PATH) as f:
        return yaml.safe_load(f) or {}


def _save_data(data: dict) -> None:
    with open(ROLES_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------
# Role hierarchy
# ------------------------------------------------------------------

def get_hierarchy() -> dict:
    """Return the role → accessible-roles mapping."""
    return _load_data().get("hierarchy", _DEFAULT_HIERARCHY)


# ------------------------------------------------------------------
# User management
# ------------------------------------------------------------------

def list_users() -> dict:
    """Return {user_id: role} for all configured users."""
    return _load_data().get("users", {})


def get_user_role(user_id: str) -> str:
    """Return the single assigned role for a user (defaults to 'public')."""
    return list_users().get(user_id, "public")


def get_user_roles(user_id: str) -> list[str]:
    """
    Return all roles the user is allowed to query, expanded by hierarchy.

    Example:
        get_user_roles("alice")  # alice is admin
        → ["admin", "analyst", "public"]
    """
    role = get_user_role(user_id)
    return get_hierarchy().get(role, ["public"])


def set_user_role(user_id: str, role: str) -> None:
    """Add or update a user's role. Persists immediately to roles.yaml."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}.")
    data = _load_data()
    data.setdefault("users", {})[user_id] = role
    _save_data(data)


def remove_user(user_id: str) -> None:
    """Remove a user from roles.yaml."""
    data = _load_data()
    data.get("users", {}).pop(user_id, None)
    _save_data(data)
