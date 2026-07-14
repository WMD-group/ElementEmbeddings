"""Utilities for species."""

from __future__ import annotations

import re
from typing import Match


def parse_species(species: str) -> tuple[str, float]:
    """
    Parse a species string into its atomic symbol and oxidation state.

    :param species: the species string
    :return: a tuple of the atomic symbol and oxidation state

    """
    match = re.match(r"([A-Za-z]+)([0-9]*[\+\-])", species)
    if match is None:
        return _parse_species_old(species)
    ele, oxi_state = match.groups()
    if oxi_state[-1] in ["+", "-"]:
        charge = (int(oxi_state[:-1] or 1)) * (-1 if "-" in oxi_state else 1)
        return ele, float(charge)
    return ele, 0.0


def _parse_species_old(species: str) -> tuple[str, float]:
    """
    Parse a species string into its atomic symbol and oxidation state.

    :param species: the species string
    :return: a tuple of the atomic symbol and oxidation state

    """
    element_match: Match[str] | None = re.match(r"[A-Za-z]+", species)
    if element_match is None:
        msg = f"{species} is not a valid species string"
        raise ValueError(msg)
    ele = element_match.group(0)

    charge_match = re.search(r"(\d+\.\d+|\d+)", species)
    ox_state = float(charge_match.group(1)) if charge_match else 0

    if "-" in species:
        ox_state *= -1

    # Handle cases of X+ or X- (instead of X1+ or X1-)
    # as well as X0+ and X0-

    if ox_state == 0 and "0" in species:
        ox_state = 0

    elif "+" in species and ox_state == 0:
        ox_state = 1

    elif ox_state == 0 and "-" in species:
        ox_state = -1

    return ele, float(ox_state)


def get_sign(charge: float) -> str:
    """Get string representation of a number's sign.

    Args:
        charge (int): The number whose sign to derive.

    Returns:
        sign (str): either '+', '-', or '' for neutral.

    """
    if charge > 0:
        return "+"
    elif charge < 0:
        return "-"
    else:
        return ""
