from jax.lib import xla_bridge
global GLOBAL_BACKEND

print("JAX Backend: %s"%(xla_bridge.get_backend().platform))
