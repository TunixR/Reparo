"""A stand-in for `Router.get_conversation_cost`, so agent runs don't fetch prices."""


async def free_of_charge(*_args: object) -> float:
    """Price any conversation at zero, as a method of `Router` or a plain callable."""
    return 0.0
