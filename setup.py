from setuptools import setup, find_packages

setup(
    name='agentic_search',
    version='4.1.2',
    packages=find_packages(),
    install_requires=[
        'arxiv',
        'beautifulsoup4',
        'duckduckgo-search',
        'langchain-community',
        'langchain-core',
        'nltk',
        'pydantic',
        'pypdf',
        'python-dotenv',
        'requests',
        'yollama',
        'ypostgres_lib'
    ],
    author='yactouat',
    author_email='yactouat@yactouat.com',
    description='code for an agentic search tool using Langchain',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/markets-agent/agentic-search',
    license='MIT',
)
