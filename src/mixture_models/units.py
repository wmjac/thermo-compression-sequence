"""Explicit conversions into kT and nm^-3; no inferred simulation units."""
from dataclasses import dataclass
import math
import re
from scipy.constants import Boltzmann, Avogadro, elementary_charge


def normalize(unit):
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Units must be nonempty strings.")
    return (unit.strip().lower().replace("å", "angstrom").replace("Å", "angstrom")
            .replace("⁻³", "^-3").replace("³", "^3").replace("−", "-")
            .replace("·", "").replace("*", "").replace(" ", ""))


def scaled_unit(unit):
    text = normalize(unit)
    match = re.match(r"^([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(.+)$", text)
    # '1/nm^3' is a reciprocal unit, rather than a scale followed by '/nm^3'.
    if match and not match.group(2).startswith("/"):
        scale = float(match.group(1))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid unit scale: {unit!r}")
        return scale, match.group(2)
    return 1.0, text


def concentration_factor(unit):
    scale, base = scaled_unit(unit)
    factors = {
        "nm^-3": 1.0, "nm-3": 1.0, "1/nm^3": 1.0, "nm^(-3)": 1.0,
        "angstrom^-3": 1000.0, "angstrom-3": 1000.0, "1/angstrom^3": 1000.0,
        "a^-3": 1000.0, "a-3": 1000.0,
        "m^-3": 1e-27, "1/m^3": 1e-27, "um^-3": 1e-9,
        "m": Avogadro * 1e-24, "mol/l": Avogadro * 1e-24,
        "mm": Avogadro * 1e-27, "mmol/l": Avogadro * 1e-27,
        "um": Avogadro * 1e-30, "umol/l": Avogadro * 1e-30,
        "mol/m^3": Avogadro * 1e-27,
    }
    if base not in factors:
        raise ValueError(f"Unsupported concentration unit {unit!r}; see docs/units.md.")
    return scale * factors[base]


def temperature_transform(unit):
    base = normalize(unit)
    if base in ("k", "kelvin"):
        return 1.0, 0.0
    if base in ("c", "degc", "°c", "celsius"):
        return 1.0, 273.15
    if base in ("f", "degf", "°f", "fahrenheit"):
        return 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0
    raise ValueError(f"Unsupported temperature unit {unit!r}.")


def energy_factor(unit, temperature_k):
    scale, base = scaled_unit(unit)
    energy_j = {
        "j": 1.0, "kj": 1000.0, "j/mol": 1.0 / Avogadro,
        "kj/mol": 1000.0 / Avogadro, "kcal/mol": 4184.0 / Avogadro,
        "cal/mol": 4.184 / Avogadro, "ev": elementary_charge,
    }
    if base in ("kt", "kbt"):
        return scale
    if base not in energy_j:
        raise ValueError(f"Unsupported potential-energy unit {unit!r}.")
    return scale * energy_j[base] / (Boltzmann * temperature_k)


def pressure_factor(unit, temperature_k):
    scale, base = scaled_unit(unit)
    pressure_pa = {
        "pa": 1.0, "kpa": 1000.0, "mpa": 1e6, "bar": 1e5,
        "atm": 101325.0, "j/m^3": 1.0,
    }
    if base in pressure_pa:
        return scale * pressure_pa[base] * 1e-27 / (Boltzmann * temperature_k)
    for suffix in ("/nm^3", "nm^-3"):
        if base.endswith(suffix):
            return scale * energy_factor(base[:-len(suffix)], temperature_k)
    for suffix in ("/angstrom^3", "angstrom^-3", "/a^3", "a^-3"):
        if base.endswith(suffix):
            return scale * 1000.0 * energy_factor(base[:-len(suffix)], temperature_k)
    raise ValueError(f"Unsupported pressure unit {unit!r}; see docs/units.md.")


@dataclass(frozen=True)
class UnitSystem:
    declarations: dict
    temperature_k: float
    concentration: float
    pressure: float
    potential_energy: float
    temperature_scale: float
    temperature_offset: float

    @classmethod
    def from_block(cls, block):
        required = ("pressure", "potential_energy", "temperature", "concentration", "nominal_temperature")
        if not isinstance(block, dict) or any(key not in block for key in required):
            raise ValueError(f"Every data JSON requires a units block containing {required}.")
        scale, offset = temperature_transform(block["temperature"])
        nominal = block["nominal_temperature"]
        if isinstance(nominal, bool) or not isinstance(nominal, (int, float)):
            raise ValueError("nominal_temperature must be a number in the declared temperature unit.")
        temperature_k = float(nominal) * scale + offset
        if not math.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("Nominal temperature must be finite and above absolute zero.")
        return cls(dict(block), temperature_k, concentration_factor(block["concentration"]),
                   pressure_factor(block["pressure"], temperature_k),
                   energy_factor(block["potential_energy"], temperature_k), scale, offset)

    def observable(self, name, values):
        """Only physical summary fields are converted; sample counts/tau are unchanged."""
        factors = {"pressure": self.pressure, "pe": self.potential_energy,
                   "temperature": self.temperature_scale}
        if name not in factors:
            return dict(values)
        result = dict(values)
        for field in ("mean", "stddev", "sem"):
            if field in result:
                result[field] = float(result[field]) * factors[name]
        if name == "temperature" and "mean" in result:
            result["mean"] += self.temperature_offset
        return result
