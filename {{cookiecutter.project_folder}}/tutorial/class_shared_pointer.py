from {{cookiecutter.package_name}}._class_shared_pointer import create, consume

data = create()
data.name = "sphere"
data.value = 42
print(data.to_string())

consume(data)
print(data.to_string())