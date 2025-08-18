"""
pyEQL salt matching test suite
==============================

This file contains tests for the salt-matching algorithm used by pyEQL in
salt_ion_match.py
"""

import logging
import platform
from itertools import combinations, product
from typing import Final

import numpy as np
import pytest

from pyEQL.salt_ion_match import Salt
from pyEQL.solution import Solution

_CONJUGATE_BASES = [
    ("HCO3[-1]", "CO3[-2]"),
    ("H2PO4[-1]", "PO4[-3]"),
]
_CONJUGATE_BASE_PAIRS = [((("Na", 1), conjugates[0]), (("Na", 1), conjugates[1])) for conjugates in _CONJUGATE_BASES]
_OXIDATION_STATE_IONS = list(combinations([(cl_ion, -1) for cl_ion in ["Cl", "ClO", "ClO2"]], r=2))
_OXIDATION_STATE_PAIRS = [((("Na", 1), anion[0]), (("Na", 1), anion[1])) for anion in _OXIDATION_STATE_IONS]


@pytest.fixture(name="cutoff", params=[1e-8])
def fixture_cutoff(request: pytest.FixtureRequest) -> float:
    return float(request.param)


@pytest.fixture(name="use_totals", params=[True])
def fixture_use_totals(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture(name="salt_dict")
def fixture_salt_dict(solution: pyEQL.Solution, cutoff: float, use_totals: bool) -> dict[str, dict[str, float | Salt]]:
    salt_dict: dict[str, dict[str, float | Salt]] = solution.get_salt_dict(cutoff=cutoff, use_totals=use_totals)
    return salt_dict


def test_salt_init() -> None:
    s = Salt("Na[+1]", "Cl[-1]")
    assert s.formula == "NaCl"
    assert s.cation == "Na[+1]"
    assert s.anion == "Cl[-1]"
    assert s.nu_anion == 1
    assert s.nu_cation == 1
    assert s.z_cation == 1
    assert s.z_anion == -1

    s = Salt("Fe+3", "OH-1")
    assert s.formula == "Fe(OH)3"
    assert s.cation == "Fe[+3]"
    assert s.anion == "OH[-1]"
    assert s.nu_anion == 3
    assert s.nu_cation == 1
    assert s.z_cation == 3
    assert s.z_anion == -1


def test_single_salt_mono() -> None:
    """
    test matching a solution with a single monovalent salt
    """
    s1 = pyEQL.Solution([["Na+", "2 mol/L"], ["Cl-", "2 mol/L"]])
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "NaCl"
    assert s1.get_salt().cation == "Na[+1]"
    assert s1.get_salt().anion == "Cl[-1]"
    assert s1.get_salt().nu_cation == 1
    assert s1.get_salt().nu_anion == 1


def test_single_salt_di() -> None:
    """
    test matching a solution with a single divalent salt
    """
    s1 = pyEQL.Solution([["Na+", "4 mol/L"], ["SO4-2", "2 mol/L"]])
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "Na2SO4"
    assert s1.get_salt().cation == "Na[+1]"
    assert s1.get_salt().anion == "SO4[-2]"
    assert s1.get_salt().nu_cation == 2
    assert s1.get_salt().nu_anion == 1


def test_single_salt_tri() -> None:
    """
    test matching a solution with a single trivalent salt
    """
    s1 = pyEQL.Solution([["Fe+3", "1 mol/L"], ["Cl-", "3 mol/L"]])
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "FeCl3"
    assert s1.get_salt().cation == "Fe[+3]"
    assert s1.get_salt().anion == "Cl[-1]"
    assert s1.get_salt().nu_cation == 1
    assert s1.get_salt().nu_anion == 3


def test_single_salt_di_tri() -> None:
    """
    test matching a solution with a divalent cation and trivalent anion
    """
    s1 = pyEQL.Solution([["Ca+2", "3 mol/L"], ["PO4-3", "2 mol/L"]])
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "Ca3(PO4)2"
    assert s1.get_salt().cation == "Ca[+2]"
    assert s1.get_salt().anion == "PO4[-3]"
    assert s1.get_salt().nu_cation == 3
    assert s1.get_salt().nu_anion == 2


def test_single_salt_tri_di() -> None:
    """
    test matching a solution with a trivalent cation and divalent anion
    """
    s1 = pyEQL.Solution([["Fe+3", "2 mol/L"], ["SO4[-2]", "3 mol/L"]])
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "Fe2(SO4)3"
    assert s1.get_salt().cation == "Fe[+3]"
    assert s1.get_salt().anion == "SO4[-2]"
    assert s1.get_salt().nu_cation == 2
    assert s1.get_salt().nu_anion == 3


def test_single_ion() -> None:
    """
    test matching a solution containing only a single ion
    """
    # Must set pH to meet default cutoff concentration in Solution.get_salt
    s1 = pyEQL.Solution(solutes={"Fe+3": "1 mol/L"}, pH=14)
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "Fe(OH)3"
    assert s1.get_salt().cation == "Fe[+3]"
    assert s1.get_salt().anion == "OH[-1]"
    assert s1.get_salt().nu_cation == 1
    assert s1.get_salt().nu_anion == 3


@pytest.mark.skipif(platform.machine() == "arm64" and platform.system() == "Darwin", reason="arm64 not supported")
def test_salt_with_equilibration() -> None:
    """
    test matching a solution containing a salt, before and after equilibration.
    Due to speciation changes, the concentration of the salt will decrease unless
    get_salt_dict() uses total concentrations
    """
    s1 = pyEQL.Solution({"Mg+2": "1 mol/L", "Cl-": "2 mol/L"})
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "MgCl2"
    assert s1.get_salt().cation == "Mg[+2]"
    assert s1.get_salt().anion == "Cl[-1]"
    assert s1.get_salt().nu_cation == 1
    assert s1.get_salt().nu_anion == 2
    assert np.isclose(s1.get_salt_dict(cutoff=0.0)["MgCl2"]["mol"], 1)
    s1.equilibrate()
    assert s1.get_salt().formula == "MgCl2"
    assert np.isclose(s1.get_salt_dict(cutoff=0.0)["MgCl2"]["mol"], 1)


def test_salt_asymmetric() -> None:
    """
    test matching a solution where the cation and anion concentrations
    are not equal
    """
    s1 = pyEQL.Solution([["Na+", "1 mol/kg"], ["Cl-", "4 mol/kg"]])
    assert isinstance(s1.get_salt(), pyEQL.salt_ion_match.Salt)
    assert s1.get_salt().formula == "NaCl"
    assert s1.get_salt().cation == "Na[+1]"
    assert s1.get_salt().anion == "Cl[-1]"
    assert s1.get_salt().nu_cation == 1
    assert s1.get_salt().nu_anion == 1


@pytest.mark.parametrize("salts", [[]])
def test_should_return_empty_dict_for_empty_solution(salt_dict: dict[str, dict[str, float | Salt]]) -> None:
    assert not salt_dict


# The parametrizations below ensures that hydroxide is the most abundant anion and can pair with excess cations
@pytest.mark.parametrize(("salt_conc", "anion_scale", "pH"), [(0.01, 0.0, 13.0), (0.01, 0.1, 13.0)])
def test_should_match_excess_cations_with_hydroxide(
    salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
) -> None:
    base = Salt(salts[0].cation, "OH[-1]")
    assert base.formula in salt_dict


# The parametrizations below ensures that protons are the most abundant cations and can pair with excess anions
@pytest.mark.parametrize(("salt_conc", "cation_scale", "pH"), [(0.01, 0.0, 1.0), (0.01, 0.1, 1.0)])
def test_should_match_excess_anions_with_protons(
    salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
) -> None:
    acid = Salt("H[+1]", salts[0].anion)
    assert acid.formula in salt_dict


# This parametrization ensures that the concentration of the salt is higher than that of water (~55 M)
@pytest.mark.parametrize(("anion_scale", "salt_conc", "salts"), [(0.0, 100.0, [Salt("Na", "Cl")])])
def test_should_log_warning_for_high_salt_concentrations(
    solution: Solution, use_totals: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=solution.logger.name)
    _ = solution.get_salt_dict(use_totals=use_totals)
    expected_record = (
        solution.logger.name,
        logging.WARNING,
        "H2O(aq) is not the most prominent component in this Solution!",
    )
    assert expected_record in caplog.record_tuples


class TestSaltDictTypes:
    parametrizations: Final[dict[str, list[str]]] = {"ion_pair": ["basic"], "ion_pairs": ["basic"]}

    @staticmethod
    @pytest.fixture(name="cation_scale", params=[0.5, 1.0, 1.5])
    def fixture_cation_scale(request: pytest.FixtureRequest) -> float:
        return float(request.param)

    @staticmethod
    def test_should_store_mol_as_floats(salt_dict: dict[str, dict[str, float | Salt]]) -> None:
        mol_values = [d["mol"] for d in salt_dict.values()]
        mol_values_are_floats = [isinstance(mol, float) for mol in mol_values]
        assert all(mol_values_are_floats)

    @staticmethod
    def test_should_store_salt_as_salts(salt_dict: dict[str, dict[str, float | Salt]]) -> None:
        salt_values = [d["salt"] for d in salt_dict.values()]
        salt_values_are_salts = [isinstance(salt, Salt) for salt in salt_values]
        assert all(salt_values_are_salts)


class TestGetSaltDict:
    parametrizations: Final[dict[str, list[str]]] = {"ion_pair": ["basic"], "ion_pairs": ["basic"]}

    @staticmethod
    def test_should_match_equimolar_ion_equivalents(
        salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
    ) -> None:
        salts_in_dict = [salt.formula in salt_dict for salt in salts]
        assert all(salts_in_dict)

    @staticmethod
    @pytest.mark.parametrize("cation_scale", [1.01])
    def test_should_match_salts_with_excess_cation(
        salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
    ) -> None:
        salts_in_dict = [salt.formula in salt_dict for salt in salts]
        assert all(salts_in_dict)

    @staticmethod
    @pytest.mark.parametrize("anion_scale", [1.01])
    def test_should_match_salts_with_excess_anion(
        salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
    ) -> None:
        salts_in_dict = [salt.formula in salt_dict for salt in salts]
        assert all(salts_in_dict)

    @staticmethod
    def test_should_not_include_water(salt_dict: dict[str, dict[str, float | Salt]]) -> None:
        assert "HOH" not in salt_dict

    @staticmethod
    # For appropriate test coverage, ensure that "cutoff" is parametrized such that it exceeds the salt concentration
    # for at least one parametrization (see salt_conc, anion/cation_scale, salt_ratio fixtures for details on how to
    # control salt concentrations)
    # Parametrization over volume is required to ensure that concentrations and not absolute molar quantities are
    # used as a cutoff (note that volume is passed to the "solution" fixture)
    @pytest.mark.parametrize(("cutoff", "volume"), product([0.0, 0.75, 1.0, 1.5], ["0.5 L", "1 L", "2 L"]))
    def test_should_not_return_salts_with_concentration_below_cutoff(
        salt_dict: dict[str, dict[str, float | Salt]],
        cutoff: float,
        solution: Solution,
    ) -> None:
        mw = solution.get_property(solution.solvent, "molecular_weight")
        assert mw is not None
        mw = mw.to("kg/mol")
        solvent_mass = solution.components[solution.solvent] * mw.m
        salt_concentrations_above_cutoff = [d["mol"] / solvent_mass >= cutoff for d in salt_dict.values()]
        assert all(salt_concentrations_above_cutoff)

    @staticmethod
    # Analogous commentary to test_should_not_return_salts_with_concentration_below_cutoff except replace "lower" with
    # "higher"
    @pytest.mark.parametrize("salt_conc_units", ["mol/kg"])
    @pytest.mark.parametrize(("cutoff", "volume"), product([0.0, 0.75, 1.0, 1.5], ["0.5 L", "1 L", "2 L"]))
    def test_should_return_all_salts_with_concentrations_above_cutoff(
        salt_dict: dict[str, dict[str, float | Salt]],
        salts: list[Salt],
        salt_conc: float,
        salt_ratio: float,
        cutoff: float,
    ) -> None:
        expected_salts = [salt for i, salt in enumerate(salts) if salt_conc * (salt_ratio**i) >= cutoff]
        expected_salts_in_salt_dict = [salt.formula in salt_dict for salt in expected_salts]
        assert all(expected_salts_in_salt_dict)

    @staticmethod
    def test_should_calculate_correct_concentration_for_salts(
        salt_dict: dict[str, dict[str, float | Salt]],
        salts: list[Salt],
        salt_conc: float,
        salt_ratio: float,
        volume: str,
    ) -> None:
        vol, _ = volume.split()
        vol_mag = float(vol)
        expected_moles = dict.fromkeys([s.formula for s in salts], 0.0)

        for i, salt in enumerate(salts):
            expected_moles[salt.formula] += salt_conc * (salt_ratio**i) * vol_mag

        salts_have_correct_concentrations = []

        for salt, expected in expected_moles.items():
            calculated = salt_dict[salt]["mol"]
            salts_have_correct_concentrations.append(np.isclose(calculated, expected, atol=1e-16))

        assert all(salts_have_correct_concentrations)


class TestGetSaltDictMultipleSalts(TestGetSaltDict):
    @staticmethod
    @pytest.fixture(name="salts")
    def fixture_salts(ion_pairs: tuple[tuple[str, str], tuple[str, str]]) -> list[Salt]:
        major_salt_cation, major_salt_anion = ion_pairs[0]
        minor_salt_cation, minor_salt_anion = ion_pairs[1]
        major_salt = Salt(major_salt_cation, major_salt_anion)
        minor_salt = Salt(minor_salt_cation, minor_salt_anion)

        return [major_salt, minor_salt]

    @staticmethod
    @pytest.mark.parametrize("cation_scale", [1.1])
    def test_should_match_salts_with_excess_cation_if_cation_enough_for_both_anions(
        salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
    ) -> None:
        major_salt, minor_salt = salts
        mixed_salt = Salt(major_salt.cation, minor_salt.anion)
        assert major_salt.formula in salt_dict
        assert mixed_salt.formula in salt_dict or mixed_salt.formula == "HOH"

    @staticmethod
    @pytest.mark.parametrize("anion_scale", [1.1])
    def test_should_match_salts_with_excess_anion_if_anion_enough_for_both_cations(
        salts: list[Salt], salt_dict: dict[str, dict[str, float | Salt]]
    ) -> None:
        major_salt, minor_salt = salts
        mixed_salt = Salt(minor_salt.cation, major_salt.anion)
        assert major_salt.formula in salt_dict
        assert mixed_salt.formula in salt_dict or mixed_salt.formula == "HOH"

    @staticmethod
    @pytest.mark.parametrize("use_totals", [False])
    @pytest.mark.parametrize(("ion_pairs"), _CONJUGATE_BASE_PAIRS)
    def test_should_include_salt_for_low_concentration_conjugate_base_when_use_totals_false(
        salt_dict: dict[str, dict[str, float | Salt]], salts: list[Salt]
    ) -> None:
        salts_in_dict = [salt.formula in salt_dict for salt in salts]
        assert all(salts_in_dict)

    @staticmethod
    @pytest.mark.parametrize(("ion_pairs"), _CONJUGATE_BASE_PAIRS)
    def test_should_not_include_salt_for_low_concentration_conjugate_base_when_use_totals_true(
        salt_dict: dict[str, dict[str, float | Salt]],
        salts: list[Salt],
    ) -> None:
        major_salt, minor_salt = salts
        assert major_salt.formula in salt_dict
        assert minor_salt.formula not in salt_dict

    @staticmethod
    @pytest.mark.parametrize(("cation_scale", "salt_ratio", "cutoff"), [(0.9, 1.0, 0.0)])
    def test_should_order_salts_by_amount(salt_dict: dict[str, dict[str, float | Salt]]) -> None:
        salt_amounts = [d["mol"] for d in salt_dict.values()]
        assert salt_amounts == sorted(salt_amounts, reverse=True)

    @staticmethod
    @pytest.mark.parametrize("ion_pairs", _OXIDATION_STATE_PAIRS)
    def test_should_match_salts_for_different_oxidation_states_when_use_totals_is_true(
        salt_dict: dict[str, dict[str, float | Salt]], salts: list[Salt]
    ) -> None:
        salts_in_dict = [salt.formula in salt_dict for salt in salts]
        assert all(salts_in_dict)


# TODO: Add integration test: for example, with salt water
