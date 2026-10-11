# Units and physical quantities

Every physical dataset JSON supplies `units.pressure`, `units.potential_energy`,
`units.temperature`, `units.concentration`, and numeric `units.nominal_temperature`.
Declarations describe stored values; adding a block does not require rewriting
existing simulation statistics.

The temperature unit applies to both the nominal value and `observables.temperature`.
The energy unit applies to `observables.pe`. Concentration units apply to both
chain and monomer concentrations. Counts are chain counts. Pressure is total
system pressure, and LAMMPS potential energy is a total-system extensive energy.

| Quantity | Internal physical unit |
|---|---|
| Chain and monomer number density | nm^-3 |
| Pressure and free-energy density | kT nm^-3 |
| Potential energy and excess chemical potentials | kT |
| Nominal / observed temperature | K |
| Normalized pressure residual and its squared loss | dimensionless |

For temperature T in K, pressure in Pa is multiplied by `1e-27 / (k_B T)`.
Pressure in atm is first multiplied by exactly 101325 Pa/atm. At 300 K the
combined factor is approximately 0.02446313291792483; calculations use the full
constants and each file's actual nominal temperature, not this displayed number.

Energy in J is divided by k_B T. Molar energies are divided by N_A first;
one kcal/mol corresponds to 4184/N_A J. A concentration in mol/L is multiplied
by N_A * 1e-24 to obtain nm^-3. One angstrom^-3 equals 1000 nm^-3.

Supported concentration strings include `nm^-3`, `1/nm^3`, `Å^-3`, `A^-3`,
`angstrom^-3`, `m^-3`, `um^-3`, `M`, `mM`, `uM`, `mol/L`, `mmol/L`, `umol/L`,
and `mol/m^3`. Positive numeric scale prefixes such as `0.1 nm^-3` and
`1e-3 M` are supported. Concentration is a number density of chains or monomers,
not a mass concentration. Mass-based inputs need additional information and are
not guessed.

Supported pressure strings include `Pa`, `kPa`, `MPa`, `bar`, `atm`, `J/m^3`,
`kT nm^-3`, `kT/nm^3`, and supported energy units followed by `/nm^3` or
`/angstrom^3` (for example `kJ/mol/nm^3`). Potential energy strings include
`kT`, `kBT`, `J`, `kJ`, `J/mol`, `kJ/mol`, `cal/mol`, `kcal/mol`, and `eV`.
Unit strings are case-insensitive; inverse-volume Unicode superscripts are
accepted. `K`, `degC`/`°C`, and `degF`/`°F` are supported temperatures.

Means are scaled and, for temperatures, offset. Standard deviations and SEMs
receive only the corresponding multiplicative scale. Autocorrelation times,
sample counts, simulation identifiers, and header volume are retained in the
source snapshots. Header volume is not used in physical calculations because
the agreed units block does not declare its units; concentration is taken from
the explicitly unit-labelled concentration field.

The loader converts to float64 physical records immediately. Training then casts
batches to the model protocol's dtype (float32 by default). `dataset.npz` archives
the converted records at float64 precision. Missing optional observations are
represented by NaN in this binary archive; required pressure/concentration values
must be finite. JSON metadata uses strict finite JSON numbers.

## Decoder reference concentration

Let c_ref = 0.1 nm^-3, and x = sum_i c_i z_i in nm^-3. The supplied architecture
evaluates its MLP at the dimensionless input u = x/c_ref and defines

    beta f_ex(x) = c_ref F(u),
    F(u) = u.T G(u) u.

This is the repository's `x*10` and `F/10` expressed as a reference scale. It
preserves the original initialization and network parameterization. The pressure
and derivative formulas are

    beta P_ex = c_ref [u.dot(grad F(u)) - F(u)],
    grad_x(beta f_ex) = grad_u F(u),
    Hessian_x(beta f_ex) = Hessian_u F(u) / c_ref.

The model's physical boundaries use nm^-3 and kT nm^-3 throughout. Removing this
reference scale would change the validated parameterization and is not part of
the refactor.
