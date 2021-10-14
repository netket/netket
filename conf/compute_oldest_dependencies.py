import pkg_resources

package_name = "netket"

package = pkg_resources.working_set.by_key[package_name]

oldest_dependencies = []

for requirement in package.requires():
    dependency = requirement.project_name
    if requirement.extras:
        dependency += '[' + ','.join(requirement.extras) + ']'
    for comparator, version in requirement.specs:
        if comparator == '==':
            if len(requirement.specs) != 1:
                raise ValueError('Invalid dependency: {requirement}'.format(requirement=requirement))
            dependency += '==' + version
        elif comparator == '<=':
            if len(requirement.specs) != 2:
                raise ValueError('Invalid dependency: {requirement}'.format(requirement=requirement))
        elif comparator == '>=':
            dependency += '==' + version
        elif comparator == '~=':
            dependency += '==' + version
    oldest_dependencies.append(dependency)

for dependency in oldest_dependencies:
    print(dependency)

with open("oldest_requirements.txt", 'w') as f:
    for dependency in oldest_dependencies:
        f.write(f"{dependency}\n")
