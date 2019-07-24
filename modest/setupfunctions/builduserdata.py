import yaml
import numpy as np
import datetime
import os


class UserData:
    def __init__(self, **response):
        for k, v in response.items():
            if isinstance(v, dict):
                self.__dict__[k] = UserData(**v)
            else:
                self.__dict__[k] = v

        return

    def __getitem__(self, key):
        return self.__dict__[key]
    def items(self):
        return self.__dict__.items()
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def __repr__(self):
        return("UserData object %s" %self.__dict__)
    def __contains__(self, key):
        return self.__dict__.__contains__(key)
    def __iter__(self):
        return self.__dict__.__iter__()

    def toDict(self):
        newDict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, UserData):
                newDict[k] = v.toDict()
            else:
                newDict[k] = v
        return newDict

def buildUserData(yamlFile):
    with open(yamlFile) as f:
        dataMap = yaml.safe_load(f)

    return UserData(**dataMap)



