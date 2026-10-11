import json
import numpy as np
import pytest
from scipy.constants import Boltzmann, Avogadro
from mixture_models.data import load_json_files
from mixture_models.units import UnitSystem, concentration_factor, pressure_factor


def block(**changes):
    return {"pressure": "atm", "potential_energy": "kcal/mol", "temperature": "K",
            "concentration": "nm^-3", "nominal_temperature": 300.0, **changes}


def test_pressure_conversion_and_sem_use_nominal_temperature():
    units = UnitSystem.from_block(block())
    assert units.pressure == pytest.approx(101325e-27 / (Boltzmann * 300))
    observed = units.observable("pressure", {"mean": -10, "stddev": 2, "sem": 0.1, "tau_int": 17})
    assert observed["mean"] == pytest.approx(-10 * units.pressure)
    assert observed["sem"] == pytest.approx(0.1 * units.pressure)
    assert observed["tau_int"] == 17
    assert units.potential_energy == pytest.approx(4184 / (Avogadro * Boltzmann * 300))


@pytest.mark.parametrize("unit,factor", [("nm^-3", 1), ("0.1 nm^-3", 0.1), ("Å^-3", 1000),
    ("M", Avogadro * 1e-24), ("mM", Avogadro * 1e-27), ("mol/m^3", Avogadro * 1e-27)])
def test_concentration_units(unit, factor):
    assert concentration_factor(unit) == pytest.approx(factor)


def test_temperature_offset_does_not_apply_to_uncertainty():
    units = UnitSystem.from_block(block(temperature="degC", nominal_temperature=26.85))
    assert units.temperature_k == pytest.approx(300)
    observed = units.observable("temperature", {"mean": 26.9, "stddev": 0.2, "sem": 0.01})
    assert observed["mean"] == pytest.approx(300.05)
    assert observed["stddev"] == pytest.approx(0.2)
    assert observed["sem"] == pytest.approx(0.01)


def test_unknown_missing_and_dimensionally_wrong_units_fail():
    with pytest.raises(ValueError, match="units block"):
        UnitSystem.from_block(None)
    with pytest.raises(ValueError, match="Unsupported pressure"):
        UnitSystem.from_block(block(pressure="kT"))
    with pytest.raises(ValueError, match="Unsupported concentration"):
        UnitSystem.from_block(block(concentration="guess"))


def test_json_ingestion_does_not_assume_length_or_units(tmp_path):
    source = {"units": block(pressure="Pa", potential_energy="kT", concentration="M"),
              "systems": [{"directory_name": "s1", "input_data_header": {"mixture_properties": {
                  "chains": [["A" * 17, 7], ["F" * 31, 3]], "total chains": 10,
                  "chain concentration": 0.1}}, "observables": {"pressure": {"mean": 1000.0},
                                                                "pe": {"mean": 2.0}}}]}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(source))
    records, _ = load_json_files([path])
    record = records[0]
    assert record.ctot == pytest.approx(0.1 * Avogadro * 1e-24)
    assert record.concentrations == pytest.approx((record.ctot * 0.7, record.ctot * 0.3))
    assert record.pressure == pytest.approx(1000e-27 / (Boltzmann * 300))
    assert record.observables["pe"]["mean"] == 2
    assert [len(s) for s in record.sequences] == [17, 31]
