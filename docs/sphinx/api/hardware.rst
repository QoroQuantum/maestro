GPU and QuEST
=============

.. note::

   These functions are in every build, but the backends they load are not part
   of the pip package. GPU simulation requires GPU access (see
   :doc:`/getting-started/gpu-access`), and QuEST requires the separate
   ``libmaestroquest`` plugin. Without them, the ``is_*_available`` functions
   return ``False``. ``is_quest_available`` also returns ``False`` until
   ``init_quest`` has loaded the plugin. ``is_distributed_gpu_available`` and
   ``finalize_distributed_mpi_gpu`` exist in Linux builds only.

.. currentmodule:: maestro

.. autofunction:: is_gpu_available
.. autofunction:: init_gpu
.. autofunction:: get_gpu_device_count
.. autofunction:: select_gpu_device
.. autofunction:: is_quest_available
.. autofunction:: init_quest
