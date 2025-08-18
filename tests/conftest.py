"""
Test Suite Reference
====================

This description summarizes the organization of pytest fixtures for the test suite.

Solution compositions are ultimately controlled by the output of the `salts` fixture. The `salts` fixture is
parametrized in two ways. The `ion_pair` fixture enables parametrization over specific single salt compositions.
As described by the `_IonPair` type alias, this fixture returns a 2-tuple. Each element in the 2-tuple is a 2-tuple
whose elements (str, float) describe the ion's formula and charge, respectively. This fixture is then used to
parametrize the `salts` fixture. The `ion_pairs` fixture facilitates parametrization over specific
combinations of ion pairs (e.g., an ion and its conjugate acid/base).

Special parametrizations of the `ion_pair` and and `ion_pairs` fixtures can be requested for test classes by
setting the `parametrizations` class attribute to a dictionary mapping the fixture name to any number of the following
strings. (This is implemented in `pytest_generate_tests()` below.)

- "basic": This requests a parametrization that includes ion pairs which represent all combinations of
  mono/di/trivalent ions, including monoatomic and polyatomic ions.
- "pitzer": This requests a parametrization that includes ion pairs which represent *nearly* all
  combinations of mono/di/trivalent ions for which Pitzer activity and molar volume parameters exist.

Solution compositions can be further controlled using the floats corresponding to the
`salt_conc`, `cation_scale`, `anion_scale`, and `salt_ratio` fixtures.

- `salt_conc`: determines the *base* concentration of salts; if `cation_scale = anion_scale = salt_ratio = 1.0`, then
  the concentration of each ion will be that resulting from dissolving a concentration of the salt equal to `salt_conc`.
- `cation_scale`/`anion_scale`: a factor with which to scale the concentration of the cation/anion in a salt relative
  to `salt_conc`. This is applied to each pair of ions in `salts`. If `cation_scale = 0.0`/`anion_scale = 0.0`, then
  only the anions/cations are added to `solution`.
- `salt_ratio`: This is the ratio of the concentration of any given Salt in `salts` relative to the previous Salt in
  the list. Note that if this value is 0.0, then no more than one Salt is added to `solution`.
"""

from collections.abc import Iterable
from itertools import combinations, product
from typing import Any, Literal

import pytest

from pyEQL.engines import EOS
from pyEQL.salt_ion_match import Salt
from pyEQL.solution import Solution
from pyEQL.utils import standardize_formula

# cation (formula, charge), anion (formula, charge)
_IonPair = tuple[tuple[str, float], tuple[str, float]]


def create_general_salts() -> list[_IonPair]:
    anions = [("Cl", -1.0), ("SO4", -2.0), ("PO4", -3.0), ("Fe(CN)6", -3.0), ("H3(CO)2", -1.0)]
    cations = [("Na", 1.0), ("Ca", 2.0), ("Fe", 3.0), ("H4Br", 3.0), ("NH4", 1.0)]
    salts = list(product(cations, anions))
    salts += [(("H", 1.0), anion) for anion in anions[:3]]
    salts += [(cation, ("OH", -1.0)) for cation in cations[:3]]

    return salts


def create_pitzer_salts() -> list[_IonPair]:
    return [
        # One pair for each combination of mono/di/trivalent ions (data permitting)
        (("Na", 1.0), ("Cl", -1.0)),
        (("Co", 2.0), ("Cl", -1.0)),
        (("Fe", 3.0), ("Cl", -1.0)),
        (("K", 1.0), ("SO4", -2.0)),
        (("Cu", 2.0), ("SO4", -2.0)),
        (("Al", 3.0), ("SO4", -2.0)),
        (("K", 1.0), ("PO4", -3.0)),
        (("H4Br", 3.0), ("N", -3.0)),
        # More polyatomic ion combinations
        (("Ag", 1.0), ("NO3", -1.0)),
        (("Ba", 2.0), ("NO3", -1.0)),
        (("NH4", 1.0), ("Cl", -1.0)),
        (("NH4", 1.0), ("NO3", -1.0)),
        (("NH4", 1.0), ("SO4", -2.0)),
        # H+ and OH- containing pairs
        (("Na", 1.0), ("OH", -1.0)),
        (("H", 1.0), ("Cl", -1.0)),
        (("H", 1.0), ("NO3", -1.0)),
        # Miscellaneous salts in database
        (("K", 1.0), ("H3(CO)2", -1.0)),
        (("Mg", 2.0), ("H6(CO)4", -2.0)),
    ]


_ION_PAIRS = create_general_salts()
_PITZER_ION_PAIRS = create_pitzer_salts()
_TEST_SCENARIOS: dict[str, dict[str, Iterable[Any]]] = {
    "ion_pair": {"basic": _ION_PAIRS, "pitzer": _PITZER_ION_PAIRS},
    "ion_pairs": {"basic": combinations(_ION_PAIRS[:3], r=2), "pitzer": combinations(_PITZER_ION_PAIRS[:3], r=2)},
}


# Implement class-based parametrization scenarios
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if not hasattr(metafunc.cls, "parametrizations"):
        return

    for fixture, parametrizations in metafunc.cls.parametrizations.items():
        for parametrization in parametrizations:
            argvalues = _TEST_SCENARIOS[fixture][parametrization]
            ids = [repr(x) for x in argvalues]
            metafunc.parametrize(fixture, argvalues, ids=ids)


@pytest.fixture(name="ion_pair")
def fixture_ion_pair() -> _IonPair:
    return (("H", 1.0), ("OH", -1.0))


@pytest.fixture(name="cation")
def fixture_cation(ion_pair: _IonPair) -> tuple[str, float]:
    cation, _ = ion_pair
    return cation


@pytest.fixture(name="anion")
def fixture_anion(ion_pair: _IonPair) -> tuple[str, float]:
    _, anion = ion_pair
    return anion


@pytest.fixture(name="salt")
def fixture_salt(cation: tuple[str, float], anion: tuple[str, float]) -> Salt:
    return Salt(
        cation=standardize_formula(f"{cation[0]}[{cation[1]}]"),
        anion=standardize_formula(f"{anion[0]}[{anion[1]}]"),
    )


@pytest.fixture(name="salts")
def fixture_salts(salt: Salt) -> list[Salt]:
    return [salt]


# When cation_scale = anion_scale = 1.0, this is equal to the concentration of the first salt in salts
@pytest.fixture(name="salt_conc", params=[1.0])
def fixture_salt_conc(request: pytest.FixtureRequest) -> float:
    return float(request.param)


# Used to scale cation concentration relative to anion concentration
@pytest.fixture(name="cation_scale", params=[1.0])
def fixture_cation_scale(request: pytest.FixtureRequest) -> float:
    return float(request.param)


# Used to scale anion concentration relative to cation concentration
@pytest.fixture(name="anion_scale", params=[1.0])
def fixture_anion_scale(request: pytest.FixtureRequest) -> float:
    return float(request.param)


# Ratio of the concentration of each Salt in the `salts` fixture to that of the previous Salt in the list
# This must be low enough such that the ordering of cation/anion equivalents coincides with the ordering
# of the corresponding Salt objects in the `salt` fixture. Too high a ratio can make it difficult to
# predict which ions will comprise the major and minor salts.
@pytest.fixture(name="salt_ratio", params=[0.25])
def fixture_salt_ratio(request: pytest.FixtureRequest) -> float:
    return float(request.param)


@pytest.fixture(name="salt_conc_units", params=["mol/L"])
def fixture_salt_conc_units(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# This is the default fixture for parametrizing the solution fixture
@pytest.fixture(name="solutes")
def fixture_solutes(
    salts: list[Salt],
    salt_conc: float,
    cation_scale: float,
    anion_scale: float,
    salt_ratio: float,
    salt_conc_units: str,
) -> dict[str, str]:
    solute_values = {salt.anion: 0.0 for salt in salts}
    solute_values.update({salt.cation: 0.0 for salt in salts})

    for i, salt in enumerate(salts):
        # Scale salt component concentrations
        cation_conc = salt_conc * salt.nu_cation * cation_scale * (salt_ratio**i)
        anion_conc = salt_conc * salt.nu_anion * anion_scale * (salt_ratio**i)
        # Increase solute concentrations
        solute_values[salt.cation] += cation_conc
        solute_values[salt.anion] += anion_conc

    # Only include solutes with non-zero concentrations
    return {k: f"{v} {salt_conc_units}" for k, v in solute_values.items() if v}


# This is an alternative way to parametrize the solution fixture
# This fixture is preferred if specific pairs of solutes are required (e.g., salts of a conjugate acid/base pair)
@pytest.fixture(name="ion_pairs", params=combinations(_ION_PAIRS[:3], r=2))
def fixture_ion_pairs(request: pytest.FixtureRequest) -> tuple[tuple[str, str], tuple[str, str]]:
    ion_pairs: tuple[tuple[str, str], tuple[str, str]] = request.param
    return ion_pairs


@pytest.fixture(name="volume", params=["1 L"])
def fixture_volume(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(name="pH", params=[7.0])
def fixture_pH(request: pytest.FixtureRequest) -> float:
    return float(request.param)


@pytest.fixture(name="solvent", params=["H2O"])
def fixture_solvent(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(name="engine", params=["native"])
def fixture_engine(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(name="database", params=[None])
def fixture_database(request: pytest.FixtureRequest) -> str | None:
    return request.param if request.param is None else str(request.param)


@pytest.fixture(name="solution")
def fixture_solution(
    solutes: dict[str, str],
    volume: str,
    pH: float,
    solvent: str | list[Any],
    engine: EOS | Literal["native", "ideal", "phreeqc"],
    database: str | None,
) -> Solution:
    return Solution(solutes=solutes, volume=volume, pH=pH, solvent=solvent, engine=engine, database=database)


# Parameter rules for Pitzer activity/osmotic coefficients
# see May, et al. (doi:10.1021/je2009329)
@pytest.fixture(name="alphas")
def fixture_alphas(ion_pair: _IonPair) -> tuple[float, float]:
    cation, anion = ion_pair
    if cation[1] >= 2 and anion[1] <= -2:
        if cation[1] >= 3 or anion[1] <= -3:
            alpha1 = 2.0
            alpha2 = 50.0
        else:
            alpha1 = 1.4
            alpha2 = 12
    else:
        alpha1 = 2.0
        alpha2 = 0.0

    return alpha1, alpha2
