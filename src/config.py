import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class BaseConfig:

    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "apidata.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
