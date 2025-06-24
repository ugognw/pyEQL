# Getting Concentrations

## Get the amount of a specific solute

To get the amount of a specific solute, use `get_amount()` and specify the units you want:

```{eval-rst}
.. doctest::

   >>> from pyEQL import Solution
   >>> s = Solution({"Mg+2": "0.5 mol/L", "Cl-": "1.0 mol/L"})
   >>> s.get_amount('Mg[+2]', 'mol')
   <Quantity(0.5, 'mole')>

```

`get_amount` is highly flexible with respect to the types of units it can interpret. You
can request amounts in moles, mass, or equivalents (i.e., charge-weighted moles) per
unit of mass or volume.

```{eval-rst}
.. doctest::
   >>> s.get_amount('Mg[+2]', 'M')
   <Quantity(0.5..., 'molar')>
   >>> s.get_amount('Mg[+2]', 'm')
   <Quantity(0.50612..., 'mole / kilogram')>
   >>> s.get_amount('Mg[+2]', 'eq/L')
   <Quantity(1.0, 'mole / liter')>
   >>> s.get_amount('Mg[+2]', 'ppm')
   <Quantity(12152.5, 'milligram / liter')>
   >>> s.get_amount('Mg[+2]', 'ppb')
   <Quantity(12152500.0, 'microgram / liter')>
   >>> s.get_amount('Mg[+2]', 'ppt')
   <Quantity(12152..., 'nanogram / liter')>

```

```{important}
The unit `'ppt'` is ambiguous in the water community. To most researchers, it means
"parts per trillion" or ng/L, while to many engineers and operators it means "parts
per THOUSAND" or g/L. `pyEQL` interprets `ppt` as **parts per trillion**.
```

You can also request dimensionless concentrations as weight percent (`'%'`),
mole fraction (`'fraction'`) or the total _number_
of particles in the solution (`'count'`, useful for setting up simulation boxes).

```{eval-rst}
.. doctest::

   >>> s.get_amount('Mg[+2]', '%')
   <Quantity(1.17358..., 'dimensionless')>
   >>> s.get_amount('Mg[+2]', 'fraction')
   <Quantity(0.00887519..., 'dimensionless')>
   >>> s.get_amount('Mg[+2]', 'count')
   <Quantity(3.01107038e+23, 'dimensionless')>

```

## See all components in the solution

You can inspect the solutes present in the solution via the `components` attribute. This comprises a dictionary of solute formula: moles, where 'moles' is the number of moles of that solute in the `Solution`. Note that the solvent (water) is present in `components`, too.
`components` is reverse sorted, with the most predominant component (i.e., the solvent)
listed first.

```{eval-rst}
   .. doctest::

   >>> from pyEQL import Solution
   >>> s = Solution({"Mg+2": "0.5 mol/L", "Cl-": "1.0 mol/L"})
   >>> s.components
   {'H2O(aq)': 54.836..., 'Cl[-1]': 1.0, 'Mg[+2]': 0.5, 'OH[-1]': 1...e-07, 'H[+1]': 1e-07}

```

Similarly, you can use the properties `anions`, `cations`, `neutrals`, and `solvent` to
retrieve subsets of `components`:

```{eval-rst}
.. doctest::

   >>> s.anions
   {'Cl[-1]': 1.0, 'OH[-1]': 1...e-07}
   >>> s.cations
   {'Mg[+2]': 0.5, 'H[+1]': 1e-07}
   >>> s.neutrals
   {'H2O(aq)': 54.836...}
   >>> s.solvent
   'H2O(aq)'

```

Like `components`, all of the above dicts are sorted in order of decreasing amount.

## Salt vs. Solute Concentrations

Sometimes the concentration of a dissolved _salt_ (e.g., MgCl2) is of greater interest
than the concentrations of the individual solutes (Mg+2 and Cl-). `pyEQL` has the
ability to interpret a `Solution` composition and represent it as a mixture of salts.

To retrieve only _the predominant salt_ (i.e., the salt with the highest concentration),
use `get_salt`. This returns a `Salt` object with several useful attributes.

```{eval-rst}
.. doctest::

   >>> from pyEQL import Solution
   >>> s = Solution({"Mg+2": "0.4 mol/L", "Na+": "0.1 mol/L", "Cl-": "1.0 mol/L"})
   >>> s.get_salt()
   <pyEQL.salt_ion_match.Salt object at ...>
   >>> s.get_salt().formula
   'MgCl2'
   >>> s.get_salt().anion
   'Cl[-1]'
   >>> s.get_salt().z_cation
   2.0
   >>> s.get_salt().nu_anion
   2

```

To see a `dict` of all the salts in given solution, use `get_salt_dict()`. This method
returns a dict keyed by the salt's formula, where the values are `Salt` objects converted
into dictionaries. All the usual attributes like `anion`, `z_cation` etc. are accessible
in the corresponding keys. Each value also contains a `mol` key giving the moles
of the salt present.

```{eval-rst}
.. doctest::

   >>> from pyEQL import Solution
   >>> s = Solution({"Mg+2": "0.4 mol/L", "Na+": "0.1 mol/L", "Cl-": "1.0 mol/L"})
   >>> s.get_salt_dict()
   {'MgCl2': {'@module': 'pyEQL.salt_ion_match',
            '@class': 'Salt', '@version': '...',
            'cation': 'Mg[+2]',
            'anion': 'Cl[-1]',
            'mol': 0.4},
   'NaCl': {'@module': 'pyEQL.salt_ion_match',
            '@class': 'Salt', '@version': '...',
            'cation': 'Na[+1]',
            'anion': 'Cl[-1]',
            'mol': 0.1},
   'NaOH': {'@module': 'pyEQL.salt_ion_match',
            '@class': 'Salt', '@version': '...',
            'cation': 'Na[+1]',
            'anion': 'OH[-1]',
            'mol': 1...e-07}}

```

Refer to the [Salt Matching module reference](internal.md#salt-matching-module) for more
details.

## Total Element Concentrations

"Total" concentrations (i.e., concentrations of all species containing a particular
element) are important for certain types of equilibrium calculations. These can
be retrieved via `get_total_amount`. `get_total_amount` takes an element name as
the first argument, and a unit as the second.

```{eval-rst}
.. doctest::

   >>> from pyEQL import Solution
   >>> s = Solution({"Mg+2": "0.5 mol/L", "Cl-": "1.0 mol/L"})
   >>> s.equilibrate()
   >>> s.components
   {'H2O(aq)': 54.83680464152719, 'Cl[-1]': 0.918668..., 'Mg[+2]': 0.418668..., 'MgCl[+1]': 0.081331..., 'OH[-1]': 1.467944...e-07, 'H[+1]': 1.18339...e-07, 'HCl(aq)': 1.2388705...e-08, 'MgOH[+1]': 3.97474...e-13, 'O2(aq)': 7.02712...e-25, 'HClO(aq)': 1.554487...e-27, 'ClO[-1]': 6.33936...e-28, 'H2(aq)': 5.7925...e-35, 'ClO2[-1]': 0.0, 'ClO3[-1]': 0.0, 'ClO4[-1]': 0.0, 'HClO2(aq)': 0.0}
   >>> s.get_total_amount('Mg', 'mol')
   <Quantity(0.499..., 'mole')>

```

## Elements present in a `Solution`

If you just want to know the elements present in the `Solution`, use `elements`. This
returns a list of elements, sorted alphabetically.

```{eval-rst}
.. doctest::

   >>> from pyEQL import Solution
   >>> s = Solution({"Mg+2": "0.5 mol/L", "Cl-": "1.0 mol/L"})
   >>> s.elements
   ['Cl', 'H', 'Mg', 'O']

```
