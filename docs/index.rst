.. meta::
   :description lang=en: NetKet is an open-source project delivering cutting-edge methods
                 for the study of many-body quantum systems with artificial neural networks
                 and machine learning techniques. NetKet provides state-of-the-art Neural-Network
                 Quantum states, and advanced learning algorithms to find the ground-state of many-body Hamiltonians.
                 NetKet provides a modular infrastructure for the development and application of machine-learning
                 techniques to many-body quantum systems. You can set up your custom many-body Hamiltonian,
                 observables, lattices, and machines in minutes.
                 The learning algorithms used in NetKet are intrinsically amenable to massive parallelism.
                 NetKet is built using MPI primitives, and can scale up to thousands of CPU cores.
                 Find out our challenges and get involved in the NetKet Project now.
                 Contributing developers will author a paper describing the NetKet library.
                 NetKet is supported by the Simons Foundation and the Flatiron Institute.
                 NetKet’s developer lead and founder is Giuseppe Carleo.


.. title:: NetKet

.. container:: jumbotron jumbotron-fluid landing-page-box landing-page-box-logo

    .. rst-class:: h1

      **NetKet**

    .. rst-class:: h2

      **Machine Learning for Many-Body Quantum Systems**

    .. raw:: html

      <div class="btn-group">
        <a class="btn btn-primary btn-lg" href="https://www.sciencedirect.com/science/article/pii/S2352711019300974" role="button">Paper</a>
        <div class="btn-group">
          <a class="btn btn-info btn-lg" href="getting_started.html" role="button">pip install netket</a>
        </div>
      </div>



.. container:: centered-bubble

    .. container:: jumbotron-fluid centered-bubble-header
      
        .. rst-class:: h3

            What is Netket?

        .. rst-class:: text-muted

            NetKet is an open-source project delivering cutting-edge methods for the study of many-body quantum systems with artificial neural networks and machine learning techniques.


.. container:: centered-bubble

    .. container:: jumbotron-fluid centered-bubble-header
      
        .. rst-class:: h3

            23 august 2021: NetKet 3 ❤️ Jax

        .. rst-class:: text-muted

            18 months in development, NetKet 3.0 indicates a major new step for the NetKet project. 
            NetKet has been totally rewritten in Python and is now a `Jax <https://jax.readthedocs.io>`-based library. 
            This guarantees outstanding performance while allowing researchers to exploit
            machine-learning frameworks to define advanced Neural-Networks. 
            
            GPUs and Google's TPUs are now supported too!

            Update now and try the new `examples <https://github.com/netket/netket/tree/master/Examples>`!



.. panels::
    :container: container-fluid pb-3
    :header: text-center text-bold fas card-frontpage-header
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-4 

    ----
    :header: + fa-university 

    Neural Quantum States
    ^^^^^^^^^^^^^^^^^^^^^

    NetKet provides state-of-the-art Neural-Network Quantum states, and advanced learning algorithms to find the ground-state of many-body Hamiltonians.

    ---
    :header: + fa-graduation-cap

    Easy to Learn
    ^^^^^^^^^^^^^

    NetKet has a library of simple Neural Quantum states and sensible defaults, allowing you to get started with as few as 10 lines of code. 

    ---
    :header: + fa-cogs

    Highly Customizable
    ^^^^^^^^^^^^^^^^^^^^

    NetKet provides a modular infrastructure for the development and application of machine-learning techniques to many-body quantum systems. You can set up your custom many-body Hamiltonian, observables, lattices, and machines in minutes.

    ---
    :header: + fa-project-diagram 

    Interoperable
    ^^^^^^^^^^^^^

    Netket is based on `Jax <https://jax.readthedocs.io>`_, therefore you can use any Neural Network Architecture written in one of the several Jax Frameworks, such as haiku or flax.

    ---
    :header: + fa-server

    Run in Parallel
    ^^^^^^^^^^^^^^^^

    The learning algorithms used in NetKet are intrinsically amenable to massive parallelism. NetKet is built using MPI primitives, and can scale up to thousands of CPU cores.


    ---
    :header: + fa-group 

    Collaborative Effort
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    NetKet wants to be a common platform for the development of new algorithms to study the most challenging open problems in many-body quantum physics. Building upon a set of well-tested primitives and on a solid infrastructure, researchers can get publication-grade results in less time.


.. toctree::
   :maxdepth: 2
   :caption: Reference Documentation
   :hidden:

   docs/getting_started
   docs/changelog
   docs/superop
   docs/varstate
   docs/sr
   docs/drivers
   docs/sharp-bits


.. toctree::
   :maxdepth: 2
   :caption: Extending NetKet
   :hidden:

   docs/custom_models
   docs/custom_preconditioners
   docs/custom_expect

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation
   :hidden:

   docs/contributing
   docs/writing-tests


.. toctree::
   :maxdepth: 3
   :caption: API documentation
   :hidden:

   docs/api-stability
   docs/api
