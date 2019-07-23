#######################################
Custom Substates and Signals
#######################################

In this section, we'll go over the steps required to write a new substate and signal object.  The problem we'll consider is two-dimensional position estimation.

***************************************
Writing a Substate Object
***************************************
As discussed in previous sections, the substate object is the object which stores some component of the state vector, or possibly the entire state vector, depending on the application.  Part of the utility of the modest package is that it allows users to quickly write their own substates, and then combine those states in a modular way.  In this section, we'll go through the details of what is required to write a substate object.

Required Methods
=======================================
In order to function as a substate, there are seven methods which a class must have.  These methods are outlined below.  

- :meth:`__init__`
  Initialize the object
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

  Of these methods, only the last two (timeUpdate and getMeasurementMatrices) are nescessarily specific to substate.  The others are often not unique to the substate. Consequently, these functions have been defined in the base class :class:`~modest.substates.substate.SubState`.  So, we can inherit from :class:`~modest.substates.substate.SubState` and only have to write the :meth:`timeUpdate` and :meth:`getMeasurementUpdateMatrices` methods.  We can do that as follows.

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
           matrixKey = self.objectID + ' ' + source.signalID() + ' range'
           HDict[matrixKey] = H
           RDict[matrixKey] = R
           dYDict[matrixKey] = dY

       if 'bearingAngle' in measurement:
           H = np.array([[
               positionDifference[1]/np.square(predictedRange),
               0,
               positionDifference[0]/np.square(predictedRange),
               0
           ]])
            
           R = np.array([[measurement['bearingAngle']['var']]])

           dY = measurement['bearingAngle']['value'] - H.dot(currentStateVector)
       
           matrixKey = self.objectID + ' ' + source.signalID() + ' bearingAngle'
            
           HDict[matrixKey] = H
           RDict[matrixKey] = R
           dYDict[matrixKey] = dY
            
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

Again, as with the dynamics dictionary, the measurement may contain quantities of interest to the substate along with irrelevant quantities that might be of interest to other, unrelated substates.  It is the job of the :meth:`getMeasurementUpdateMatrices` method to evaluate what quantities are contained in the measurement, and how to build the appropriate measurement matrices.  This is what is happening in the if statements.

As written, the substate only cares about two types of measurements: range and bearing angle.  :meth:`getMeasurementUpdateMatrices` check whether either (or both) of these types of measurements are  contained in the measurement dictionary, and if so, it builds the appropriate measurement update matrices.  The mathematics of the measurement update matrices are not particularly relevant to the mechanics of the modest package, so we won't worry about the derivations here. (Of course, your equations must be correct for the estimator to function properly!)

It is important to note the output of :meth:`getMeasurementUpdateMatrices`.  Specifically, the output should be three dictionaries: one corresponding to the measurement mapping matrix (often denoted as :math:`H`), one corresponding to the measurement noise matrix (often denoted as :math:`R`), and one corresponding to the measurement residual matrix (often detnoted as :math:`\delta y`).  In each of these dictionaries, modest will expect to find key-value pairs, where each key contains a unique label corresponding to the measurement component, and the value corresponds to the sub-component of the measurement matrix that corresponds to that measurement.  For example:

.. code-block:: python

   HDict = {
       'object1 range': rangeHMatrix
       'object1 bearingAngle': bearingAngleHMatrix
   }

   RDict = {
       'object1 range': rangeRMatrix
       'object1 bearingAngle': bearingAngleRMatrix
   }
   
   dYDict = {
       'object1 range': rangedYMatrix
       'object1 bearingAngle': bearingAngledYMatrix
   }
   

Mathematically, these components will be joined together to form the following matrices.



.. math::
   \mathbf{H} = \begin{bmatrix}\textrm{obj. 1 range } H \textrm{ matrix}\\\textrm{obj. 1 bearing } H \textrm{ matrix} \\ \textrm{(other sub }H\textrm{ matrices)}\end{bmatrix}
      
.. math::
   \mathbf{R} = \begin{bmatrix}\textrm{obj. 1 range } R \textrm{ matrix} & 0 & \vdots \\ 0 &\textrm{obj. 1 bearing } R \textrm{ matrix} & \vdots \\ \cdots & \cdots & \textrm{(other sub }R\textrm{ matrices)}\end{bmatrix}

.. math::
   \boldsymbol{\delta} \mathbf{Y} = \begin{bmatrix}\textrm{obj. 1 range } \delta Y \textrm{ matrix} \\ \textrm{obj. 1 bearing } \delta Y \textrm{ matrix} \\ \textrm{(other sub }\delta Y\textrm{ matrices)}\end{bmatrix}

The reason for packaging the measurement update matrices in this way is to allow them to be assembled on the fly by the estimator, and combined as needed with other measurement matrices.  It is important that the key that is associated with each measurement sub-matrix be unique not only to this substate, but across all possible substate/signal combinations.  A combination of the substate name (a unique identifier), the signal ID (also a unique identifier), and the measurement type will ensure this.



***************************************
Writing a Signal Object
***************************************
While substates are the objects which manage subcomponents of the state vector, signal objects are the objects that manage the signal model.  Any type of signal that you want to use as a measurement of a substate should be modeled as a signal object.

Required Methods
=======================================
As with substates, there are a few methods which a signal object *must* have if it is going to function as a signal object.  These methods are outlined below.  

- :meth:`__init__`
  Initialize the object
- :meth:`signalID`
  Returns the signal's ID
- :meth:`computeAssociationProbability`
  Computes a non-normalized probability that a given measurement originated from the signal source

And again, as with substates, there is a base class :class:`~modest.signals.signalsource.SignalSource` which handles some of the boiler-plate code.  We begin our class definition as follows:

.. code-block:: python
		  
   import modest as md
   import numpy as np
   from scipy.linalg import block_diag
   class rangeAndBearing(md.signals.signalsource.SignalSource):


It turns out that both of these methods are defined 
Initialization Method
=======================================

.. code-block:: python   

   def __init__ (self, stateObjectID, position=[0,0]):
       self.position=position
       self.stateObjectID = stateObjectID
       super().__init___()
       return

This initialization method is pretty rudimentary.  Again, this is in part because the base class :class:`~modest.signals.signalsource.SignalSource` does some of the work for us.  There are two extra bits that we had to take care of.

The first bit of house-keeping is the storing of the state object ID.  As currently implemented, each signal source corresponds to a signal associated with a given substate.  So, for instance, if you have two objects tracked by a single radar station, you need two signal sources: one to represent a measurement of the first object and one to represent a measurement of the second object.  If you had two radar stations, you'd need four signal sources, and so on.  Thus, the stateObjectID is the identifier by which the substate associated with this signal source can be located.

.. note::
   This requirement that signal sources be uniquely associated with one state is probably not nescessary. I will look into removing it in a later version.

The second house-keeping item is the storing of the "position" attribute.  For the type of signal we wish to model, the position from which the measurement is an essential bit of information needed to compute the measurement update.  We allow the user to set this position from the beginning.

Of course, any other attributes associated with the signal source could be stored during the initialization function as well.


Association Probability Method
=======================================
The other method which must be defined is the association probability.  It should be noted from the outset that this method is only used in some cases.  If, for instance, you're trying to implement a basic EKF in which you can uniquely associate signals with the correct sources (for instance radar tracking of aircraft with transponders), then you could get away without writing this method.  The association probability method is only required for cases in which the data association is uncertain.  Modest won't complain if you try to add a signal that doesn't have this method; it will complain if you try to do an update that requires this method.

.. code-block:: python

    def computeAssociationProbability(self, measurement, stateDict, validationThreshold=0):
        myMeasMat = stateDict[self.objectID]['stateObject'].getMeasurementMatrices(measurement, source=self)
        dY = None
        R = None
        H = None
        for key in myMeasMat['dY']:
            if H is None:
                H = myMeasMat['H'][key]
                R = myMeasMat['R'][key]
                dY = myMeasMat['dY'][key]
            else:
                H = np.vstack([H, myMeasMat['H'][key]])
                R = block_diag(R, myMeasMat['R'][key])
                dY = np.append(dY, myMeasMat['dY'][key])

        if dY is not None:
            P = stateDict[self.objectID]['stateObject'].covariance()
            Pval = P.convertCovariance('covariance').value
            # if P.form == 'cholesky':
            #     Pval = P.value.dot(P.value.transpose())
            # elif P.form == 'covariance':
            #     Pval = P.value
            # else:
            #     raise ValueError('Unrecougnized covariance specifier %s' %P.form)
            S = H.dot(Pval).dot(H.transpose()) + R

            myProbability = mvn.pdf(dY, cov=S)
        else:
            myProbability = 0
        return myProbability
