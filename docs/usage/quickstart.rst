Quick Start
=============================================


The following is a quick guide to help you get modest up and running.

Modest is designed to help you different types of estimation algorithms on a variety of problems while having to re-write as little code as possible.  The following is a basic example which showcases a few of modest's capabilities.  This guide assumes that you are at least somewhat familier with state estimation theory.  If not, there are a number of excellent books and online resources on the topic.

=================
Initialize Filter
=================

First, we need to import modest.

>>> import modest as md

The main class in modest is the :class:`~modest.modularfilter.ModularFilter` class.  This is the class that performs all of the filtering algorithms and the class that we interact the most with.  We can initialize it as follows.

>>> myFilter = md.ModularFilter()

There are a few optional arguments for initialization of :class:`~modest.modularfilter.ModularFilter`, but in general the defaults are fine.


=================
Create Substates
=================

Right now, ``myFilter`` is just a filter; it doesn't have any states it's trying to estimate or any signals it's filtering.  We want to change this.  Let's add a one-dimensional position estimation state.

>>> positionSubstate1 = md.substates.oneDimensionalPositionandVelocity()
>>> myFilter.addStates('object1', positionSubstate1)

Here, we did two things.  First, we initialized a :class:`~modest.modularfilter.ModularFilter` object.  Then, we passed that object to myFilter and told it that we are going to be estimating a state with name "object1".  The name we choose for the state doesn't matter too much, as long as it's unique (:class:`~modest.modularfilter.ModularFilter` will raise an error if you try to give it two substates of the same name).

The :class:`~modest.modularfilter.ModularFilter` class is agnostic about what kinds of states it's estimating.  This means that myFilter doesn't care what *kind* of state it's estimating.  All it cares about for now is the the substate's value, its covariance, and the substate's dimension.  It is also important to note that, while the :class:`~modest.modularfilter.ModularFilter` class doesn't care what type of substate it receives, the substate object does need to have a few essential member functions.  These member functions are prototyped in the :class:`~modest.substates.substate.SubState` base class.  An object doesn't have to inherit from :class:`~modest.substates.substate.SubState` to be added as a substate, but this can be helpful in ensuring that the object has all of the functionality needed to function as a sub-state.

Typical estimation algorithms consist of two kinds of updates; time updates and measurement updates.  We'll cover how to do both of these in this guide.  However, before we can do measurement updates, we have to tell ``myFilter`` about what kinds of signals we expect to see.

=================
Create Signals
=================

For a general estimation problem, we may have many sources of measurement for a given state.  For instance, an aircraft might measure its position in three dimensions using a combination of ground-based measurements such as VOR, GPS, and altimeter measurements.  Each of these signal sources has a different measurement model.  This information is contained in the signals object.

Signals and substates are related to each other: the substate needs to have a measurement model for each possible signal source.  On the other hand, signal sources are responsible for being able to compute their probability of association.  We can add a signal to myFilter as follows.

>>> myRangeSignal1 = md.signals.oneDObjectMeasurement('object1')
>>> myFilter.addSignalSource(myRangeSignal1, 'rangeSignal1')

Again, we did two things.  We first created a signal source, ``myRangeSignal1``.  We then added that signal source to myFilter.

As with substates, the :class:`~modest.modularfilter.ModularFilter` class is agnostic about what type of signal is being added.  It is up to the substate and signal objects to ensure that they have the nescessary functionality.  For the most part, this interaction is handled directly between the substates and signals; the :class:`~modest.modularfilter.ModularFilter` does not need to know "how the sausages are made" (i.e. it doesn't need to know the internal workings of the interactions between the signals and substates that occur).  However, as with substates, there are some member functions that the signal source *must* have in order to function as a signal source.  These are prototyped in the :class:`~modest.signals.signalsource.SignalSource` base class. An object doesn't *have* to inherit from :class:`~modest.signals.signalsource.SignalSource` to function as a signal source, but having this inheritance ensures that the proper methods exist.

Now that we've added our signals and substates, we can proceed to do estimation. :class:`~SignalSource`

=================
Time Update
=================

In general, :class:`~modest.modularfilter.ModularFilter` assumes that a substate object has all the information it needs to handle its own dynamics updates, given dynamics information.  For example, the :class:`oneDimensionalPositionandVelocity` object can formulate its own time-update matrices given elapsed time and some sort of dynamics information (like acceleration).

In order to do a time-update, we must first get the dynamics information in the format that :class:`~modest.modularfilter.ModularFilter` expects.  Dynamics information is passed in :class:`dict` format. An example is given below.

>>> myDynamics = {
>>>   'acceleration':
>>>     {'value': -9.81, 'var': 0.01},
>>> }

The dynamics dictionary contains sub-dictionaries for each quantity.  In turn those sub-dictionaries contain values and variances.  The names of the quantities can be anything, as long as the sub-state knows what to look for.  We can then perform a time-update as follows:

>>> myFilter.timeUpdateEKF(1.0, dynamics=myDynamics)

This command tells myFilter that all the substates (in this case, just ``positionSubstate1``) are being propagated forward in time by 1.0 time units (generally seconds, but could be whatever we want as long as we're consistent).  It then passes the dynamics and dT to the substates (calling :meth:`modest.substates.substate.timeUpdate`), and expects them to generate time-update matrices.  Once those time-update matrices are received, the state vector and state covariance are updated.

We note here that the substate objects are expected to have a method called ``timeUpdate``.  This is one of the methods which is prototyped in the :class:`~modest.substates.substate.SubState` class.  However, we stress that this is **only** a prototype; in the :class:`~modest.substates.substate.SubState` class, it doesn't have any functionality.  It merely serves as a "placeholder," but it is up to the developer of a specific substate class to write the actual code for the time update.  

=================
Measurement Update
=================

The other type of update in state estimation is the measurement update.  As with time-updates, :class:`~modest.modularfilter.ModularFilter` leaves it up to the individual substates to formulate their measurement matrices.  In our example, ``positionSubstate1`` is expected to be able to return measurement matrices given the measurement quantity and the measurement origin.

To do a measurement update, we need to package the measurement information in the proper format, as with time updates.  An example is given below.

>>> myMeasurement = {
>>>   'position':
>>>     {'value': -4.904, 'var': 0.1},
>>> }

Note the similarities between the dynamics dictionary and the measurement dictionary: each quantity is given by a sub-dictionary containing the value and the variance.  As with dynamics, the labels don't matter to :class:`~modest.modularfilter.ModularFilter`, as long as the labels are what the substates and signal sources are expecting.

We can then do a measurement update as follows.

>>> myFilter.measurementUpdateEKF(myMeasurement, 'rangeSignal1')

This tells the filter that we just measured the quantities given in ``myMeasurement``, and we know that they came from the signal source ``rangeSignal1``.  The filter then passes this information to each of the substates (calling :meth:`modest.substates.substate.getMeasurementUpdateMatrices`), and expects those substates to return the appropriate measurement matrices.  The filter then updates the state vector using the standard EKF equations.  The measurement is assumed to have occured at the current time, so it's the user's responsibility to perform time-updates to update the state to the measurement time before trying to do a measurement update.

Again, we note that the substate objects are expected to have a method specific method, called ``getMeasurementUpdateMatrices`` which is expected to have a specific functionality.  This is another one of the methods which is prototyped in the :class:`~modest.substates.substate.SubState` class.  As with the ``timeUpdate`` method, the :meth:`modest.substates.substate.getMeasurementUpdateMatrices` method is only a prototype or "place-holder"; it is up to the developer to program the measurement model specific to the substate and measurement source.

=================
Next Steps
=================
So far we have walked through the basic, bare-bones functionality of the ``modest`` package.  At this point, we've introduced a lot of complexity, which really isn't needed for a basic Kalman filter (which can be written in about 5 lines of code).  However, there's a lot more that the ``modest`` package can do, and the framework we've introduced here will serve to make the more complex algorithms much more managable.

.. toctree::
   :hidden:
   
   self

	     
