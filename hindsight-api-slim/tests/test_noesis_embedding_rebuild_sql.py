"""SQL contract checks for the Noesis embedding rebuild."""

from hindsight_api.engine.retain.noesis_embedding_rebuild import (
    _active_ep_count,
    _atoms_cutover_update,
    _atoms_page,
)


def test_rebuild_atom_sql_uses_current_a_status_contract():
    statements = (
        _atoms_page("noesis_core", False),
        _atoms_page("noesis_core", True),
        _active_ep_count("noesis_core"),
        _atoms_cutover_update("noesis_core", False),
        _atoms_cutover_update("noesis_core", True),
    )
    assert all("status = 'A'" in sql for sql in statements)
    assert all("status = 'active'" not in sql for sql in statements)
