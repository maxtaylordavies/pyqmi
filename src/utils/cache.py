from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_project_api import (
    EcephysProjectWarehouseApi,
)
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine


def get_cache(manifest_path, session_id=None):
    cache = EcephysProjectCache(
        manifest=manifest_path,
        fetch_api=EcephysProjectWarehouseApi(
            RmaEngine(scheme="http", host="api.brain-map.org", timeout=3600)  # 1 hour
        ),
    )

    if session_id == None:
        return cache
    else:
        return [cache, cache.get_session_data(session_id)]
