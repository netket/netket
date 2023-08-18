import pkg_resources
import toml

project = toml.load("pyproject.toml")["project"]
requirements = [pkg_resources.Requirement(pkg) for pkg in project["dependencies"]]

oldest_dependencies = []

for requirement in requirements:
    dependency = requirement.project_name
    if requirement.extras:
        dependency += "[" + ",".join(requirement.extras) + "]"
    for comparator, version in requirement.specs:
        if comparator == "==":
            if len(requirement.specs) != 1:
                raise ValueError(f"Invalid dependency: {requirement}")
            dependency += "==" + version
        elif comparator == "<=":
            if len(requirement.specs) != 2:
                raise ValueError(f"Invalid dependency: {requirement}")
        elif comparator == ">=":
            dependency += "==" + version
        elif comparator == "~=":
            dependency += "==" + version
    oldest_dependencies.append(dependency)

for dependency in oldest_dependencies:
    print(dependency)

with open("oldest_requirements.txt", "w") as f:
    for dependency in oldest_dependencies:
        f.write(f"{dependency}\n")
