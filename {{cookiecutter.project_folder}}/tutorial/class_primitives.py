from {{cookiecutter.package_name}}.class_primitives import Data

# Different ways to create Data objects
sphere = Data("sphere", 42)           # Regular constructor
cube = Data.from_name("cube")        # Using just a name
cylinder = Data.from_values("cylinder", 99)  # Explicit values

# Print out all objects
print(f"Sphere: {sphere.name}, {sphere.value}")
print(f"Cube: {cube.name}, {cube.value}")
print(f"Cylinder: {cylinder.name}, {cylinder.value}")

# Modify some properties
cylinder.name = "cone"
cylinder.value = 77
print(f"\nAfter modification: {cylinder.to_string()}")

