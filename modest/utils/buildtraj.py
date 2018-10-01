import yaml
from pypet import Environment, cartesian_product
import numpy as np
import datetime
import os
from .. import substates

## @fun buildPulsarCorrelationSubstate builds an correlation substate based on imported Traj
def buildPulsarCorrelationSubstate(
        traj,
        pulsarObject,
        processNoiseScaleFactor=None
):
    # Import and initialize values for correlation filter
    processNoise = (
        traj.correlationFilter.processNoise.value
    )  # Unitless??

    if processNoiseScaleFactor is not None:
        processNoise = processNoise * processNoise
    nFilterTaps = traj.correlationFilter.filterTaps.value
    measurementNoiseScaleFactor = (
        traj.correlationFilter.measurementNoiseScaleFactor.value
    )
    peakLockThreshold = (
        traj.correlationFilter.peakLockThreshold.value
    )

    # Now initialize the correlation substate
    correlationSubstate = substates.CorrelationVector(
        pulsarObject,
        nFilterTaps,
        pulsarObject.pulsarPeriod/(nFilterTaps+1),
        signalTDOA=0,
        TDOAVar=pulsarObject.pulsarPeriod,
        measurementNoiseScaleFactor=measurementNoiseScaleFactor,
        processNoise=processNoise,
        centerPeak=True,
        peakLockThreshold=peakLockThreshold,
    )
    return correlationSubstate




## @fun buildEnvironment creates a pypet envorinment based on a YAML input file
def buildEnvironment(yamlFile):
    # Read initialization file (assumed to be YAML)
    with open(yamlFile) as f:
        dataMap = yaml.safe_load(f)

    createDirectory = dataMap['environment'].pop('createDirectory')
    if createDirectory:
        dateTimeString = (
            datetime.date.today().strftime('%Y_%m_%d_') +
            datetime.datetime.now().strftime('%Hh%Mm%Ss')
        )
        newDir = dataMap['environment']['filename'] + '/' + dateTimeString
        os.mkdir(dataMap['environment']['filename'] + '/' + dateTimeString)
        dataMap['environment']['filename'] = (
            newDir + '/' +
            dataMap['environment']['trajectory'] +
            '.hdf5'
        )
        dataMap['environment']['add_time'] = False
    else:
        dataMap['environment']['add_time'] = True
        
    # Initialize Environment with any paramters in the environment branch of initalization file
    env = Environment(**dataMap['environment'])

    # Add the rest of the branches as parameters
    parameterDict = dataMap['parameters']
    for key in parameterDict:
        addParameterGroup(env.trajectory, key, parameterDict[key])

    exploreParameters = dataMap['exploreParameters']
    exploreDict = {}

    # In this loop, we initialize the cartesian product to be explored in Pypet, based on the exploreParameters section of the yaml file.
    for key, param in exploreParameters.items():
        # First, we figure out the range type (i.e. linear or log)
        modifiedKey = key + '.value'
        if 'rangeType' in param:
            rangeType = param['rangeType']
        else:
            rangeType = 'linear'

        if rangeType == 'linear':
            product = np.linspace(param['start'], param['stop'], param['number']).tolist()
        elif rangeType == 'log':
            product = np.logspace(param['start'], param['stop'], param['number']).tolist()
        print(product)
        exploreDict[modifiedKey] = [
            type(env.trajectory.parameters[modifiedKey])(element)
            for element in product
        ]
    env.traj.f_explore(
        cartesian_product(
            exploreDict
        )
    )
    
    
    return env, dataMap


## @fun addParameterGroup adds parameters to a pypet trajectory
#
# @details addParameterGroup adds a single parameter or group of parameters to
# a pypet trajectory.  If a dictionary is received, then the function adds a
# sub-group of parameters, then recursively adds each element of the
# dictionary to the newly created sub-group.
#
# @param traj The pypet trajectory to which the parameters are to be added
# @param name A string containing the name of the new parameter or parameter
# group
# @param parameter A single value or dict of values containing the new
# parameter(s)
#
# @return Returns the modified trajectory
def addParameterGroup(traj, name, parameter):

    # If the received parameter is a dictionary, then create a parameter
    # sub-group and recursively add each dictionary element
    if isinstance(parameter, dict):
        if 'comment' in parameter:
            comment=parameter['comment']
        else:
            comment='No comment included'
        newGroup = traj.f_add_parameter_group(name=name, comment=comment)
        for key in parameter:
            if key is not 'comment':
                addParameterGroup(newGroup, key, parameter[key])
    else:
        traj.f_add_parameter(name, parameter)

    return traj
