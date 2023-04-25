from setuptools import find_packages, setup


def get_version():
    version_file = "Swim_Fish/version.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


if __name__ == "__main__":
    # install_requires = parse_requirements()
    setup(
        name="Swim_Fish",
        version=get_version(),
        description="Fish_mmdet",
        author="Caijt",
        author_email="caijiatong@westlake.edu.cn",
        packages=find_packages(exclude=("configs", "cocostuffapi", "data", "work_dirs", "tools", "Pipeline", "thirdparty")),
        install_requires=None,
        include_package_data=True,
        ext_modules=[],
        zip_safe=False,
    )