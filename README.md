## [Flask API Servier]

Simple [Flask API] enhanced with SqlAlchemy persistence and deployment scripts via Docker - Provided by **Amal bedoui**. It has the ready-to-use route for **model1** .



<br />

## ✨ Quick Start in `Docker`

> Get the code

```bash
$ git clone 'https://github.com/amalbed/ml-test'
$ cd ml-test
```

> Start the app in Docker

```bash
$ docker-compose up --build  
```

The API server will start using the PORT `5000`.



## ✨ Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Modules](#modules)
4. [Testing](#testing)

<br />

## ✨ How to use the code

> **Step #1** - Clone the project

```bash
$ git clone 'https://github.com/amalbed/ml-test'
$ cd ml-test
```

<br />

> **Step #2** - create virtual environment using python3 and activate it (keep it outside our project directory)

```bash
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate

```

<br />

> **Step #3** - Install dependencies in virtualenv

```bash
$ pip install -r requirements.txt
```

<br />

> **Step #4** - setup `flask` command for my app

```bash
$ export FLASK_APP=run.py
$ export FLASK_ENV=development
```



<br />

> **Step #5** - start test APIs server at `localhost:5000`

```bash
$ flask run
```

<br />

> **Step #6** - download model1 from this link:`https://drive.google.com/file/d/1JGU2xJfUmW84t_xJJ79p-ZhuEEVA-wY2/view?hl=en`

```
add it to ml-test/src
```


<br />

## ✨ Project Structure

```bash
ml-test/
├── classification_reports/
│   └── classification_model_1.json
├── data/
│   ├── dataset_multi.csv
│   └── dataset_single.csv
├── nginx/
│   └── flask_api.conf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── feature_extractor.py
│   ├── main.py
│   ├── model_manager.py
│   ├── models.py
│   └── routes.py
├── Dockerfile
├── README.md
├── requirements.txt
├── run.py
└── tests.py
```

<br />

## ✨ API


> **Predict Feature** - `api/predict` (**POST** request)

```
POST api/predict
Content-Type: application/json

{
    "id":"",
    "smile":"any smile", 
    
}
```

<br />



<br />

## ✨ Testing

Run tests using `pytest tests.py`

<br />

