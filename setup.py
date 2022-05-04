from setuptools import setup


def readme():
    with open("README.md") as f:
        README = f.read()
    return README


setup(
    name="yolov5-lite",
    version="0.0.1",
    description="Yolov5-Lite: Minimal YoloV5 Implementation",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kadirnar/",
    author="Kadir Nar",
    author_email="kadir.nar@hotmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["yolov5-lite"],
    include_package_data=True,
    install_requires="requirements.txt",
)