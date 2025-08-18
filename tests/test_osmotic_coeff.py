"""
pyEQL osmotic coefficient test suite
============================================

This file contains tests for the osmotic coefficient method
employed by pyEQL.

NOTE: generally, these tests check the module output against experimental
data rather than the theoretical result of the respective functions.
"""

from typing import Final

import numpy as np
import pytest
from pint import Quantity

import pyEQL.activity_correction as ac
from pyEQL import ureg
from pyEQL.salt_ion_match import Salt
from pyEQL.solution import Solution


def test_osmotic_pressure() -> None:
    """
    The osmotic pressure of seawater is approximately 27 atm
    """
    # TODO - at present this test is inaccurate because in the complex matrix
    # of seawater, pyEQL falls back to using an ideal solution model with
    # unit osmotic coefficient.
    empty = Solution()
    assert np.isclose(empty.osmotic_pressure.to("atm").magnitude, 0, atol=1e-5)
    sea = Solution.from_preset("seawater")
    assert np.isclose(sea.osmotic_pressure.to("atm").magnitude, 27, rtol=0.15)


class Test_osmotic_pitzer:
    """
    test osmotic coefficient based on the Pitzer model
    ------------------------------------------------

    """

    parametrizations: Final[dict[str, list[str]]] = {"ion_pair": ["pitzer"]}

    @staticmethod
    @pytest.fixture(name="expected_osmotic_coefficient")
    def fixture_expected_osmotic_coefficient(
        solution: Solution,
        salt: Salt,
        salt_conc: float,
        salt_conc_units: str,
        alphas: tuple[float, float],
    ) -> Quantity:
        concentration = ureg.Quantity(salt_conc, salt_conc_units)
        param = solution.get_property(salt.formula, "model_parameters.activity_pitzer")
        assert param is not None
        return ac.get_osmotic_coefficient_pitzer(  # type: ignore[no-any-return, no-untyped-call]
            solution.ionic_strength,
            concentration,
            alphas[0],
            alphas[1],
            ureg.Quantity(param["Beta0"]["value"]).magnitude,
            ureg.Quantity(param["Beta1"]["value"]).magnitude,
            ureg.Quantity(param["Beta2"]["value"]).magnitude,
            ureg.Quantity(param["Cphi"]["value"]).magnitude,
            salt.z_cation,
            salt.z_anion,
            salt.nu_cation,
            salt.nu_anion,
            str(solution.temperature),
        )

    @pytest.mark.parametrize("salt_conc", [0.0, 1e-4, 0.1, 1.0])
    def test_dimensionality(self, solution: Solution) -> None:
        osmotic_coefficient = solution.get_osmotic_coefficient()
        assert isinstance(osmotic_coefficient, Quantity)
        assert osmotic_coefficient.dimensionality == ""
        assert osmotic_coefficient >= 0

    def test_osmotic_pitzer_ammoniumnitrate(self) -> None:
        """
        calculate the osmotic coefficient at each concentration and compare
        to experimental data for ammonium nitrate

        References
        ----------
        May, P. M., Rowland, D., Hefter, G., & Königsberger, E. (2011).
        A Generic and Updatable Pitzer Characterization of Aqueous Binary Electrolyte Solutions at 1 bar and 25 °C.
        Journal of Chemical & Engineering Data, 56(12), 5066-5077. doi:10.1021/je2009329

        """
        # list of concentrations to test, mol/kg
        conc_list = [0.25, 0.5, 0.75, 1, 1.5, 2]

        # list of published experimental osmotic coefficients
        pub_osmotic_coeff = [0.86, 0.855, 0.83, 0.825, 0.80, 0.78]

        for i, conc in enumerate(conc_list):
            solute_conc = str(conc) + "mol/kg"
            sol = Solution()
            sol.add_solute("NH4+", solute_conc)
            sol.add_solute("NO3-", solute_conc)
            result = sol.get_osmotic_coefficient()
            expected = pub_osmotic_coeff[i]

            assert np.isclose(result, expected, rtol=0.05)

    def test_osmotic_pitzer_coppersulfate(self) -> None:
        """
        calculate the osmotic coefficient at each concentration and compare
        to experimental data for copper sulate

        References
        ----------
        May, P. M., Rowland, D., Hefter, G., & Königsberger, E. (2011).
        A Generic and Updatable Pitzer Characterization of Aqueous Binary Electrolyte Solutions at 1 bar and 25 °C.
        Journal of Chemical & Engineering Data, 56(12), 5066-5077. doi:10.1021/je2009329

        """
        # list of concentrations to test, mol/kg
        conc_list = [0.25, 0.5, 0.75, 1]

        # list of published experimental osmotic coefficients
        pub_osmotic_coeff = [0.5, 0.485, 0.48, 0.485, 0.5]

        for i, conc in enumerate(conc_list):
            solute_conc = str(conc) + "mol/kg"
            sol = Solution()
            sol.add_solute("Cu+2", solute_conc)
            sol.add_solute("SO4-2", solute_conc)
            result = sol.get_osmotic_coefficient()
            expected = pub_osmotic_coeff[i]

            assert np.isclose(result, expected, rtol=0.05)

    @staticmethod
    # parametrize with single (1) many-component mixture of low concentrations
    @pytest.mark.parametrize(("salt_conc", "salt_conc_units"), [(1e-11, "mol/kg")])
    def test_should_return_unity_for_low_concentration_solutes(solution: Solution) -> None:
        assert solution.get_osmotic_coefficient().m == 1.0

    @staticmethod
    @pytest.mark.parametrize("salt_conc_units", ["mol/kg"])
    def test_should_return_osmotic_coefficient_of_major_salt_when_parameters_exist(
        solution: Solution, expected_osmotic_coefficient: Quantity
    ) -> None:
        assert solution.get_osmotic_coefficient().m == expected_osmotic_coefficient.m


# Redefine/-parametrize engine as ideal-only
class TestOsmoticIdeal:
    @staticmethod
    @pytest.fixture(name="salt")
    def fixture_salt() -> Salt:
        return Salt("Na[+1]", "Cl[-1]")

    @staticmethod
    @pytest.fixture(name="engine")
    def fixture_engine() -> str:
        return "ideal"

    @staticmethod
    def test_should_return_unit_activity_for_ideal_engine(solution: Solution) -> None:
        assert solution.get_osmotic_coefficient().m == 1.0

    @pytest.mark.parametrize("salt_conc", [0.0, 1e-4, 0.1, 1.0])
    def test_dimensionality(self, solution: Solution) -> None:
        osmotic_coefficient = solution.get_osmotic_coefficient()
        assert isinstance(osmotic_coefficient, Quantity)
        assert osmotic_coefficient.dimensionality == ""
