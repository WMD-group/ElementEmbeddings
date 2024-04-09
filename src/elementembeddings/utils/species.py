"""Utilities for species."""
import re
from typing import Tuple


def parse_species(species: str) -> Tuple[str, int]:
    """
    Parse a species string into its atomic symbol and oxidation state.

    :param species: the species string
    :return: a tuple of the atomic symbol and oxidation state

    """
    try:
        ele, oxi_state = re.match(r"([A-Za-z]+)([0-9]*[\+\-])", species).groups()
        if oxi_state[-1] in ["+", "-"]:
            charge = (int(oxi_state[:-1] or 1)) * (-1 if "-" in oxi_state else 1)
            return ele, charge
        else:
            return ele, 0
    except AttributeError:
        return _parse_species_old(species)


def _parse_species_old(species: str) -> Tuple[str, int]:
    """
    Parse a species string into its atomic symbol and oxidation state.

    :param species: the species string
    :return: a tuple of the atomic symbol and oxidation state

    """
    ele = re.match(r"[A-Za-z]+", species).group(0)

    charge_match = re.search(r"\d+", species)
    ox_state = int(charge_match.group(0)) if charge_match else 0

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

    return ele, ox_state