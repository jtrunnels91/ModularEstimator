Custom Substates and Signals
=============================================

In this section, we'll go over the steps required to write a new substate and signal object.  The problem we'll consider is two-dimensional position estimation.

=============================================
Writing a Substate Object
=============================================

In order to function as a substate, there are six methods which a class must have.  These methods are outlined below.  

- :meth:`getStateVector`
  Returns the most recent estimated state vector
- :meth:`storeStateVector`
  Receives an updated state vector (time or measurement updated) and stores it.  If the substate has any additional "post-processing" that needs to be done on the substate after an update, this is the place to do it.
- :meth:`covariance`
  Returns the current covariance.  This method is only called once, when a substate is added.  After that, the :class:`~modest.modularfilter.ModularFilter` class handles the covariance matrix.
- :meth:`dimension`
  Returns the dimension of the substate  
- :meth:`timeUpdate`
  Returns the time-update matrices, given system dynamics
- :meth:`getMeasurementUpdateMatrices`
  Returns the measurement-update matrices, given the measurement
- :meth:`__init__`
  Initialize the object, including 

Of these six functions, only the last two (timeUpdate and getMeasurementMatrices) are nescessarily specific to substate.  The other four are often not unique to the substate. Consequently, these functions have been defined in the base class :class:`~modest.substates.substate.SubState`.  So, we can inherit from :class:`~modest.substates.substate.SubState` and only have to write the :meth:`timeUpdate` and :meth:`getMeasurementUpdateMatrices` methods.  We can do that as follows.  

>>> import modest as md
>>> class twoDPositionVelocity(md.substates.SubState):

Now we need to define our methods, including the initialization method.  The initialization method has to do a few things.  It needs to get the initial estimates for the state vector and covariance matrix, and since it's inheriting from :class:`~modest.substates.substate.SubState`, it needs to initialize the "super" class.

>>> def __init__ (self, stateVector, covariannce, objectID='', time=0):
>>>    stateVectorHistory = {'stateVector': stateVector, 'covariance': covariance, 't': time, 'stateVectorID':0}
>>>    self.objectID = objectID
>>>    super().__init___(stateDimension=4, stateVectorHistory=stateVectorHistory)

This initialization function is pretty rudimentary, in part because :class:`~modest.substates.substate.SubState` does a lot of the work for us (like checking dimensionality).  Of course in a more complicated substate we might have to do more initialization.

There is one key bit of initialization that we did have to do: the creation of the stateVectorHistory dictionary.  In :class:`~modest.substates.substate.SubState` type classes, the state vector is stored in a dictionary containing (at minimum) the state vector, the covariance, and a unique identifier, the stateVectorID.  By default, every state estimate over the entire life of the object is stored in a list of such dictionaries, all handled by the :class:`~modest.substates.substate.SubState` class.  This is mainly for analysis (allowing the user to look at the time history of the state), but it is also used by  :class:`~modest.substates.substate.SubState` for storing and getting the most recent versions of the state vector when needed. :class:`~modest.substates.substate.SubState` expects to receive this dictionary on initialization; if it doesn't, or if the dict doesn't have the right members, :class:`~modest.substates.substate.SubState` will throw an error.  

Now that we've defined the initialization function, we can define our time and measurement update methods.

>>> from scipy.linalg import block_diag
>>> def timeUpdate(self, dT, dynamics=None):
>>>   subF = np.array([[1, dT],[0, 1]])
>>>   F = block_diag(subF, subF)
>>>
>>>   accelerationKey = self.objectID + 'accleration'
>>>   if accelerationKey in dynamics:
>>>      subQ = np.array([[dT4/4, dT3/2],[dT3/2, dT2]])
>>>      Q = block_diag(subQ, subQ) * dynamics[accelerationKey]['var']
>>>   else:
>>>      Q = np.zeros([self.dimension(), self.dimension()])
>>>
>>>   return {'F': F, 'Q': Q}

The main job of this relatively simple function is to generate the time-update equations for an object in two-dimensional motion with acceleration as an input, and pass them out to the caller in a standard dictionary format.

We also check to make sure that the dynamics information we're interested in is actually contained in the dynamics dictionary.  We don't assume that our substate is the only one running in the filter; there could be others with their own accelerations.  Consequently, we build in a check to verify that the dictionary we're getting contains the acceleration we're interested in.  The method the substate uses to identify it's dynamics information is up to you as the developer (i.e. you don't have to follow the "objectID + acceleration" format).  As long as the substate knows what it should be looking for, you can use whatever key you want.

The other method that we need to define is the measurement update method.  This is generally a little bit more complicated.  That's because there are potentially a lot of different kinds of measurements we need to handle.  There could be direct measurements of position (for instance from a GPS receiver), there could be range and/or bearing measurement from known navigationl beacons, or there could be velocity measurements.  Consequently, it's often best to define individual methods to handle each possible measurement

>>> def getMeasurementUpdateMatrices
