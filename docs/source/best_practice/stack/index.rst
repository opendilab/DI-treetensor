Stack Structured Data
===============================

When the tensors form the tree structures, they are often needed \
to be stacked together, like the :func:`torch.stack` implemented \
in torch.

Stack With Native PyTorch API
-----------------------------------

Here is the common code implement with native pytorch API.

.. literalinclude:: native_api.demo.py
    :language: python
    :linenos:

The output should be like below, and the assertion statements \
can be all passed.

.. literalinclude:: native_api.demo.py.txt
    :language: text
    :linenos:

We can see that the structure process function need to be \
fully implemented (like the function ``stack``). This code \
is actually not clear, and due to hard coding, if you need \
to support more data types (such as integer), you must make \
special modifications to the function.


Stack With TreeTensor API
-------------------------------

The same workflow can be implemented with treetensor API like \
the code below.

.. literalinclude:: treetensor_api.demo.py
    :language: python
    :linenos:

The output should be like below, and the assertion statements \
can be all passed as well.

.. literalinclude:: treetensor_api.demo.py.txt
    :language: text
    :linenos:

This code looks much simpler and clearer.
