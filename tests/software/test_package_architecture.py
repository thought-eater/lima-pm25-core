import subprocess
import sys


def test_root_import_does_not_eagerly_load_result_packages_or_cli() -> None:
    script = """
import sys
import lima_pm25

unexpected = [
    name
    for name in (
        "lima_pm25.cli",
        "lima_pm25.domain",
        "lima_pm25.simulation",
        "lima_pm25.estimation",
    )
    if name in sys.modules
]
assert not unexpected, unexpected
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_result_subpackages_import_from_one_distribution() -> None:
    from lima_pm25.domain import FixedStationSite, GridDefinition
    from lima_pm25.estimation import GaussianProcessFieldReconstructor
    from lima_pm25.simulation import MeasurementRecord, PM25Transport

    assert GridDefinition.__module__ == "lima_pm25.domain.spatial"
    assert FixedStationSite.__module__ == "lima_pm25.domain.stations"
    assert PM25Transport.__module__ == "lima_pm25.simulation.transport"
    assert MeasurementRecord.__module__ == "lima_pm25.simulation.sensing"
    assert (
        GaussianProcessFieldReconstructor.__module__
        == "lima_pm25.estimation.reconstruction"
    )
