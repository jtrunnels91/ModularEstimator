Advanced Measurement Updates
=============================================

As noted in :ref:`Quickstart`, if all you want to do is implement a basic Kalman filter, ``modest`` might not be the best package for you, since it does introduce a fair amount of complication for a task that really only requires a few lines of code.  If, however, you're doing something more complicated, the ``modest`` framework can help you implement a more advanced estimation scheme using the individual pieces (substates and signal sources).

One of the big advantages of this framework is that it allows you to put together the individual pieces of your estimator, and test them to make sure they work in the simple, "ideal" scenario (like the example that was shown in :ref:`Quickstart`).  Once we verify that the sub-components work individualy, then we can use them in more complex estimation algorithms.

Here, we'll continue our example from :ref:`Quickstart` to show some of the capabilities of the ``modest`` package.

Suppose we have a scenario similar to that in :ref:`Quickstart`, with a single object moving in one dimension, and we want to estimate its position.  We initialize our filter the same way:

>>> import modest as md
>>> myFilter = md.ModularFilter()
>>> positionSubstate1 = md.substates.oneDimensionalPositionandVelocity()
>>> myFilter.addStates('object1', positionSubstate1)

Now, suppose that we have two possible sources of range measurements.  We can add two signal sources as follows.

>>> myRangeSignal1 = md.signals.oneDObjectMeasurement(position=0)
>>> myRangeSignal2 = md.signals.oneDObjectMeasurement(position=100)
>>> myFilter.addSignalSource(myRangeSignal1, 'rangeSignal1')
>>> myFilter.addSignalSource(myRangeSignal2, 'rangeSignal2')

We can perform time updates just as before.

>>> myDynamics = {
>>>   'acceleration':
>>>     {'value': -9.81, 'var': 0.01},
>>> }
>>> myFilter.timeUpdateEKF(1.0, dynamics=myDynamics)

Now, suppose we have a range measurement, but we don't know from which signal source it originated.

>>> myMeasurement = {
>>>   'position':
>>>     {'value': -4.904, 'var': 0.1},
>>> }

We can't use :meth:`~modest.modularfilter.ModularFilter.measurementUpdateEKF`, because this method needs to be told what signal source the measurement originates from.  However, ``modest`` includes other measurement update techniques.  One option would be a maximum likelihood (ML) estimator.  We can perform an ML measurement without any extra work: all of the nescessary framework is already implemented.

>>> myFilter.measurementUpdateML(myMeasurement)

This command tells the filter to compute the conditional probability of association for both signal sources, then update the state estimate using the associaton with the highest probability.

At this point, a lot is happening "under the hood" that we don't see.  The substate objects are computing update matrices, the signal objects are computing association probabilities, and the :class:`~modest.modularfilter.ModularFilter` object is perfoming the global state update.  All of this happens with relatively little input from the user, because these functionalities have been built into these objects.

If the functionalities you want happen to be included in the modest package, then your job is done.  However, most likely you will want to do state estimation with different states, different signal sources, and different filtering algorithms.  The modest package is designed to make it as easy as possible to include whatever custom functionality you want to include with as little boiler-plate code as possible.  In the next section we'll show a simple example of building signal and substate objects from scratch.




