#######################################
Custom Substates and Signals
#######################################

In this section, we'll go over the steps required to write a new substate and signal object.  The problem we'll consider is two-dimensional position estimation.

***************************************
Writing a Substate Object
***************************************


Required Methods
=======================================
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

.. code-block:: python   
   import modest as md
   import numpy as np
   from scipy.linalg import block_diag
   class twoDPositionVelocity(md.substates.SubState):


Now we need to define our methods, including the initialization method.  The initialization method has to do a few things.  It needs to get the initial estimates for the state vector and covariance matrix, and since it's inheriting from :class:`~modest.substates.substate.SubState`, it needs to initialize the "super" class.

Initialization Method
=======================================

.. code-block:: python   

   def __init__ (self, stateVector, covariance, objectID='', time=0):
       stateVectorHistory = {'stateVector': stateVector, 'covariance': covariance, 't': time, 'stateVectorID':0}
       self.objectID = objectID
       super().__init___(stateDimension=4, stateVectorHistory=stateVectorHistory)

This initialization function is pretty rudimentary, in part because :class:`~modest.substates.substate.SubState` does a lot of the work for us (like checking dimensionality).  Of course in a more complicated substate we might have to do more initialization.

There is one key bit of initialization that we did have to do: the creation of the stateVectorHistory dictionary.  In :class:`~modest.substates.substate.SubState` type classes, the state vector is stored in a dictionary containing (at minimum) the state vector, the covariance, and a unique identifier, the stateVectorID.  By default, every state estimate over the entire life of the object is stored in a list of such dictionaries, all handled by the :class:`~modest.substates.substate.SubState` class.  This is mainly for analysis (allowing the user to look at the time history of the state), but it is also used by  :class:`~modest.substates.substate.SubState` for storing and getting the most recent versions of the state vector when needed. :class:`~modest.substates.substate.SubState` expects to receive this dictionary on initialization; if it doesn't, or if the dict doesn't have the right members, :class:`~modest.substates.substate.SubState` will throw an error.  

Now that we've defined the initialization function, we can define our time and measurement update methods.

Time Update Method
=======================================

.. code-block:: python   

    from scipy.linalg import block_diag
    def timeUpdate(self, dT, dynamics=None):
      subF = np.array([[1, dT],[0, 1]])
      F = block_diag(subF, subF)
   
      accelerationKey = self.objectID + 'accleration'
      if accelerationKey in dynamics:
         subQ = np.array([[dT4/4, dT3/2],[dT3/2, dT2]])
         Q = block_diag(subQ, subQ) * dynamics[accelerationKey]['var']
      else:
         Q = np.zeros([self.dimension(), self.dimension()])
   
      return {'F': F, 'Q': Q}

The main job of this relatively simple function is to generate the time-update equations for an object in two-dimensional motion with acceleration as an input, and pass them out to the caller in a standard dictionary format.

We also check to make sure that the dynamics information we're interested in is actually contained in the dynamics dictionary.  We don't assume that our substate is the only one running in the filter; there could be others with their own accelerations.  Consequently, we build in a check to verify that the dictionary we're getting contains the acceleration we're interested in.  The method the substate uses to identify it's dynamics information is up to you as the developer (i.e. you don't have to follow the "objectID + acceleration" format).  As long as the substate knows what it should be looking for, you can use whatever key you want.

Measurement Update Method
=======================================

The other method that we need to define is the measurement update method.  This is generally a little bit more complicated.  That's because there are potentially a lot of different kinds of measurements we need to handle.  There could be direct measurements of position (for instance from a GPS receiver), there could be range and/or bearing measurement from known navigational beacons, or there could be velocity measurements.  Consequently, the measurement update method can be a bit complex, and it is often convenient to define additional methods to handle individual sub-cases (this is entirely up to the user of course).

.. code-block:: python

   def getMeasurementMatrices(self, measurement, source=None):

       if not source:
           return

       HDict = {}
       RDict = {}
       dYDict = {}
        
       currentStateVector = self.stateVectorHistory[-1]['stateVector']
       currentX = currentStateVector[0]
       currentY = currentStateVector[2] 
       measurementPosition = source.position
       positionDifference = np.array([currentX, currentY]) - measurementPosition
       predictedRange = np.linalg.norm(positionDifference)
       
       if 'range' in measurement:

           H = np.array([[
               positionDifference[0] / predictedRange,
               0,
               positionDifference[1] / predictedRange,
               0
           ]])

           R = np.array([[measurement['position']['var']]])

           dY = measurement['position']['value'] - H.dot(currentStateVector)
            
           HDict[self.objectID + ' position'] = H
           RDict[self.objectID + ' position'] = R
           dYDict[self.objectID + ' position'] = dY

       if 'bearingAngle' in measurement:
           H = np.array([[
               positionDifference[1]/np.square(predictedRange),
               0,
               positionDifference[0]/np.square(predictedRange),
               0
           ]])
            
           R = np.array([[measurement['bearingAngle']['var']]])

           dY = measurement['bearingAngle']['value'] - H.dot(currentStateVector)
            
           HDict[self.objectID + ' bearingAngle'] = H
           RDict[self.objectID + ' bearingAngle'] = R
           dYDict[self.objectID + ' bearingAngle'] = dY
            
       return {'H': HDict, 'R': RDict, 'dY': dYDict}


There is a lot going on in this method, so let's unpack it a little bit at a time.  First, note the inputs.  Any time the :meth:`getMeasurementUpdateMatrices` method is called, the method expects to receive as arguments the measurement itself, as well as some kind of information about the signal source.  (At some point during development, I envisioned instances where this method would be called when no signal source information was present, so this was left as an optional argument.  However I don't think there are currently any actual implementations where this is the case).

Next, we note that the source object is arbitrary.  It is up to the :meth:`getMeasurementUpdateMatrices` method to evaluate what the signal source is, and how to generate the appropriate measurement update matrices.  In this simple example, the only checking done is to evaluate whether the signal source exists at all, but in a more complicated implimentation more checking might be nescessary.

As with the dynamics dictionary in the time update method, the measurement matrix method expects to receive the measurement as a dictionary.  A measurement may have multiple components as well.  For instance, a measurement passed to this method might look something like this:

.. code-block:: python

   measurementDict = {
       'range': {
           'value': 14.5,
           'var': 0.1
       }
       'bearingAngle': {
           'value': 0.345,
           'var': 0.001
       },
       'temperature': {
           'value': 22.4,
           'var': 2.1
       }
   }

Again, as with the dynamics dictionary, the measurement may contain quantities of interest to the substate along with irrelevant quantities that might be of interest to other, unrelated substates.

The substate then proceeds with 
