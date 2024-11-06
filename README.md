
# Data-minning



## Installation

Conda:
```bash
conda create --name data-minning --file conda_requirements.txt
```

Pip:
```bash
python3 -m venv .venv
cd .venv/Scripts/activate
pip install -r pip_requirements.txt
```
    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

Page load and HTTP 429 prevention:

`TIME_BETWEEN_LOAD_MORE` - [float] time in seconds between 'Load More' click

`TIME_FOR_PAGE_LOAD` - [float] time in seconds for page load


## On environment update

```bash
conda list -e > conda_requirements.txt
pip list --format=freeze > pip_requirements.txt
```

## Authors

- Michał Kaniowski [@mkaniowski](https://www.github.com/mkaniowski)
- Konrad Marcjanowicz [@kmarcjanowicz](https://github.com/KMarcjanowicz)
- Łukasz Kochańczyk [@LKochan123](https://github.com/LKochan123)
- Łukasz Kołodziej [@Lkolod](https://github.com/Lkolod)

