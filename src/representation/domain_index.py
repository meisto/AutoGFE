# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Sat 12 Feb 2022 04:32:25 AM CET
# Description: -
# ======================================================================
from typing import List, Set

_domains: List[Set[str]] = []


def add_domain(domain: List[str]):
    """
        Add a domain to the registry.
    """
    _domains.append(domain)

def get_other_domain_members(member: Set[str]):
    """
        Retrieve a domain by a member. Returns empty set when not member in
        any domain.
    """

    # search in domains for a domain in which it is member
    for d in _domains:
        if member in d: return d

    # Otherwise return the empty set
    return set()




