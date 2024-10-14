.. index:: ! contract

.. _contracts:

##########
Contracts
##########

Contracts in Solidity are similar to classes in object-oriented languages. They
contain persistent data in state variables, and functions that can modify these
variables. Calling a function on a different contract (instance) will perform
an EVM function call and thus switch the context such that state variables
in the calling contract are
inaccessible. A contract and its functions need to be called for anything to happen.
There is no "cron" concept in Ethereum to call a function at a particular event automatically.

.. include:: contracts/creating-contracts.rst

.. include:: contracts/visibility-and-getters.rst

.. include:: contracts/function-modifiers.rst

.. include:: contracts/transient-storage.rst

.. include:: contracts/constant-state-variables.rst
.. include:: contracts/functions.rst

.. include:: contracts/events.rst
.. include:: contracts/errors.rst

.. include:: contracts/inheritance.rst

.. include:: contracts/abstract-contracts.rst
.. include:: contracts/interfaces.rst

.. include:: contracts/libraries.rst

.. include:: contracts/using-for.rst