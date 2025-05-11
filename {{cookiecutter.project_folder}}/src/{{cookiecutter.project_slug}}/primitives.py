from {{cookiecutter.project_slug}} import _primitives  # The actual C++ module

def add(a, b):
    """Add two numbers together."""
    return _primitives.add(a, b)
