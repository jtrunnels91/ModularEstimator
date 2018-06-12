import yaml
from pypet import Environment

## @fun buildEnvironment creates a pypet envorinment based on a YAML input file
def buildEnvironment(yamlFile):
    # Read initialization file (assumed to be YAML)
    with open(yamlFile) as f:
        dataMap = yaml.safe_load(f)

    # Initialize Environment with any paramters in the environment branch of initalization file
    env = Environment(**dataMap['environment'])

    # Add the rest of the branches as parameters
    parameterDict = dataMap['parameters']
    for key in parameterDict:
        addParameterGroup(env.trajectory, key, parameterDict[key])
    
    return dataMap, env


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
