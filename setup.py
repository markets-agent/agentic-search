from setuptools import setup, find_packages

setup(
    name='agentic_search',
    version='9.1.0',
    packages=find_packages(),
    install_requires=[
        'aiocache',
        'arxiv',
        'asyncio',
        'beautifulsoup4',
        'duckduckgo-search',
        'langchain-community',
        'langchain-core',
        'langchain-openai',
        'langgraph',
        'nltk',
        'pydantic',
        'pypdf',
        'python-dotenv',
        'redis',
        'selenium',
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
