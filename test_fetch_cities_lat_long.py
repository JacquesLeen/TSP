"""
Test Suite for TSP Solver

Run tests with:
    pytest test_fetch_cities_lat_long.py -v
    pytest test_fetch_cities_lat_long.py -v --cov=fetch_cities_lat_long
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from click.testing import CliRunner


# Import functions from your script
from fetch_cities_lat_long import (
    cities_from_args,
    create_cities_dataframe,
    tsp,
    main,
    cli,
)


# ============================================
# FIXTURES
# ============================================


@pytest.fixture(name="sample_cities_df")
def fixture_sample_cities_df():
    """
    Fixture providing a sample DataFrame with test cities.

    Returns
    -------
    pd.DataFrame
        DataFrame with 3 cities and their coordinates
    """
    return pd.DataFrame(
        {
            "city": ["Boston", "Miami", "Denver"],
            "latitude": [42.3601, 25.7617, 39.7392],
            "longitude": [-71.0589, -80.1918, -104.9903],
        }
    )


@pytest.fixture
def mock_geocode_location():
    """
    Fixture providing a mock geocode location object.

    Returns
    -------
    Mock
        Mock object with latitude and longitude attributes
    """
    mock_location = Mock()
    mock_location.latitude = 40.7128
    mock_location.longitude = -74.0060
    return mock_location


@pytest.fixture(name="cli_runner")
def fixture_cli_runner():
    """
    Fixture providing a Click CLI test runner.

    Returns
    -------
    CliRunner
        Click test runner for invoking CLI commands
    """
    return CliRunner()


# ============================================
# UNIT TESTS - cities_from_args()
# ============================================


class TestCitiesFromArgs:
    """Test suite for the cities_from_args function."""

    def test_single_city(self):
        """Test with a single city argument."""
        result = cities_from_args("Boston")
        assert result == ["Boston"]
        assert isinstance(result, list)

    def test_multiple_cities(self):
        """Test with multiple city arguments."""
        result = cities_from_args("Boston", "Miami", "Denver")
        assert result == ["Boston", "Miami", "Denver"]
        assert len(result) == 3

    def test_no_cities(self):
        """Test with no arguments."""
        result = cities_from_args()
        assert result == []
        assert isinstance(result, list)

    def test_cities_with_spaces(self):
        """Test with city names containing spaces."""
        result = cities_from_args("New York", "Los Angeles", "San Francisco")
        assert result == ["New York", "Los Angeles", "San Francisco"]
        assert len(result) == 3


# ============================================
# UNIT TESTS - create_cities_dataframe()
# ============================================


class TestCreateCitiesDataframe:
    """Test suite for the create_cities_dataframe function."""

    @patch("fetch_cities_lat_long.geopy.geocoders.Nominatim")
    def test_default_cities(self, mock_nominatim):
        """Test creating DataFrame with default US cities."""
        # Mock the geocode response
        mock_geolocator = Mock()
        mock_location = Mock()
        mock_location.latitude = 40.7128
        mock_location.longitude = -74.0060
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator

        result = create_cities_dataframe()

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["city", "latitude", "longitude"]
        assert len(result) == 10  # Default has 10 cities

        # Check data types
        assert result["city"].dtype == object
        assert pd.api.types.is_numeric_dtype(result["latitude"])
        assert pd.api.types.is_numeric_dtype(result["longitude"])

    @patch("fetch_cities_lat_long.geopy.geocoders.Nominatim")
    def test_custom_cities(self, mock_nominatim):
        """Test creating DataFrame with custom cities."""
        mock_geolocator = Mock()
        mock_location = Mock()
        mock_location.latitude = 42.3601
        mock_location.longitude = -71.0589
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator

        custom_cities = ["Boston", "Miami"]
        result = create_cities_dataframe(cities=custom_cities)

        assert len(result) == 2
        assert list(result["city"]) == ["Boston", "Miami"]

    @patch("fetch_cities_lat_long.geopy.geocoders.Nominatim")
    def test_geocoding_called_correctly(self, mock_nominatim):
        """Test that geocoding is called with correct parameters."""
        mock_geolocator = Mock()
        mock_location = Mock()
        mock_location.latitude = 40.7128
        mock_location.longitude = -74.0060
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator

        create_cities_dataframe(cities=["Boston"])

        # Verify Nominatim was initialized correctly
        mock_nominatim.assert_called_with(user_agent="tsp_pandas")

        # Verify geocode was called
        mock_geolocator.geocode.assert_called_with("Boston", timeout=None)

    @patch("fetch_cities_lat_long.geopy.geocoders.Nominatim")
    def test_empty_cities_list(self, mock_nominatim):
        """Test with an empty cities list."""
        mock_geolocator = Mock()
        mock_location = Mock()
        mock_location.latitude = 40.7128
        mock_location.longitude = -74.0060
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator

        result = create_cities_dataframe(cities=[])

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)


# ============================================
# UNIT TESTS - tsp()
# ============================================


class TestTSP:
    """Test suite for the tsp function."""

    def test_tsp_returns_tuple(self, sample_cities_df):
        """Test that tsp returns a tuple of (distance, route)."""
        distance, route = tsp(sample_cities_df)

        assert isinstance(distance, float)
        assert isinstance(route, list)

    def test_tsp_route_length(self, sample_cities_df):
        """Test that route contains all cities."""
        route = tsp(sample_cities_df)[1]

        assert len(route) == len(sample_cities_df)

    def test_tsp_route_contains_all_cities(self, sample_cities_df):
        """Test that route contains each city exactly once."""
        route = tsp(sample_cities_df)[1]

        assert set(route) == set(sample_cities_df["city"].tolist())

    def test_tsp_distance_positive(self, sample_cities_df):
        """Test that calculated distance is positive."""
        distance = tsp(sample_cities_df)[0]

        assert distance > 0

    def test_tsp_distance_reasonable(self, sample_cities_df):
        """Test that distance is within reasonable bounds."""
        distance = tsp(sample_cities_df)[0]

        # Distance should be reasonable for US cities (not too small, not too large)
        assert 1000 < distance < 50000  # km

    def test_tsp_with_two_cities(self):
        """Test TSP with minimum number of cities (2)."""
        df = pd.DataFrame(
            {
                "city": ["Boston", "Miami"],
                "latitude": [42.3601, 25.7617],
                "longitude": [-71.0589, -80.1918],
            }
        )

        distance, route = tsp(df)

        assert len(route) == 2
        assert distance > 0

    @patch("fetch_cities_lat_long.shuffle")
    def test_tsp_shuffle_called(self, mock_shuffle, sample_cities_df):
        """Test that shuffle is called to randomize cities."""
        tsp(sample_cities_df)

        # Verify shuffle was called
        assert mock_shuffle.called


# ============================================
# UNIT TESTS - main()
# ============================================


class TestMain:
    """Test suite for the main function."""

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_returns_tuple(self, mock_tsp, mock_create_df, sample_cities_df):
        """Test that main returns correct tuple structure."""
        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        result = main(10, verbose=False)

        assert isinstance(result, tuple)
        assert len(result) == 3
        best_distance, best_route, stats = result
        assert isinstance(best_distance, float)
        assert isinstance(best_route, list)
        assert isinstance(stats, dict)

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_stats_structure(self, mock_tsp, mock_create_df, sample_cities_df):
        """Test that stats dictionary has correct keys."""
        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        _, _, stats = main(10, verbose=False)

        assert "best" in stats
        assert "worst" in stats
        assert "average" in stats
        assert "total_simulations" in stats

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_runs_correct_number_of_simulations(
        self, mock_tsp, mock_create_df, sample_cities_df
    ):
        """Test that main runs the specified number of simulations."""
        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        n_sims = 50
        _, _, stats = main(n_sims, verbose=False)

        assert stats["total_simulations"] == n_sims
        assert mock_tsp.call_count == n_sims

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_finds_best_distance(self, mock_tsp, mock_create_df, sample_cities_df):
        """Test that main correctly identifies the best (shortest) distance."""
        mock_create_df.return_value = sample_cities_df

        # Simulate different distances
        distances = [5000.0, 4500.0, 5200.0, 4800.0, 4300.0]  # 4300 is best
        routes = [["A", "B", "C"]] * 5
        mock_tsp.side_effect = list(zip(distances, routes))

        best_distance, _, stats = main(5, verbose=False)

        assert best_distance == 4300.0
        assert stats["best"] == 4300.0
        assert stats["worst"] == 5200.0
        assert stats["average"] == pytest.approx(4760.0)

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    @patch("fetch_cities_lat_long.click.echo")
    def test_main_verbose_mode(
        self, mock_echo, mock_tsp, mock_create_df, sample_cities_df
    ):
        """Test that verbose mode prints simulation details."""
        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        main(5, verbose=True)

        # Verify that click.echo was called (verbose output)
        assert mock_echo.called


# ============================================
# INTEGRATION TESTS - CLI Commands
# ============================================


class TestSimulateCLI:
    """Test suite for the simulate CLI command."""

    @patch("fetch_cities_lat_long.main")
    def test_simulate_default_options(self, mock_main, cli_runner):
        """Test simulate command with default options."""
        mock_main.return_value = (
            5000.0,
            ["Boston", "Miami"],
            {
                "best": 5000.0,
                "worst": 6000.0,
                "average": 5500.0,
                "total_simulations": 100,
            },
        )

        result = cli_runner.invoke(cli, ["simulate"])

        assert result.exit_code == 0
        assert "Starting 100 TSP simulations" in result.output
        assert "Simulation complete" in result.output

    @patch("fetch_cities_lat_long.main")
    def test_simulate_custom_n_simulations(self, mock_main, cli_runner):
        """Test simulate command with custom number of simulations."""
        mock_main.return_value = (
            5000.0,
            ["Boston", "Miami"],
            {
                "best": 5000.0,
                "worst": 6000.0,
                "average": 5500.0,
                "total_simulations": 500,
            },
        )

        result = cli_runner.invoke(cli, ["simulate", "--n_simulations", "500"])

        assert result.exit_code == 0
        assert "Starting 500 TSP simulations" in result.output
        mock_main.assert_called_once_with(500, verbose=False)

    @patch("fetch_cities_lat_long.main")
    def test_simulate_verbose_flag(self, mock_main, cli_runner):
        """Test simulate command with verbose flag."""
        mock_main.return_value = (
            5000.0,
            ["Boston", "Miami"],
            {
                "best": 5000.0,
                "worst": 6000.0,
                "average": 5500.0,
                "total_simulations": 100,
            },
        )

        result = cli_runner.invoke(cli, ["simulate", "--verbose"])

        assert result.exit_code == 0
        mock_main.assert_called_once_with(100, verbose=True)

    @patch("fetch_cities_lat_long.main")
    def test_simulate_with_output_file(self, mock_main, cli_runner):
        """Test simulate command with output file."""
        mock_main.return_value = (
            5000.0,
            ["Boston", "Miami"],
            {
                "best": 5000.0,
                "worst": 6000.0,
                "average": 5500.0,
                "total_simulations": 100,
            },
        )

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(cli, ["simulate", "--output", "results.txt"])

            assert result.exit_code == 0
            assert "Results saved to results.txt" in result.output

            # Check file was created
            with open("results.txt", "r", encoding="utf8") as f:
                content = f.read()
                assert "TSP SIMULATION RESULTS" in content
                assert "Best distance" in content

    @patch("fetch_cities_lat_long.main")
    def test_simulate_displays_results(self, mock_main, cli_runner):
        """Test that simulate displays correct results."""
        mock_main.return_value = (
            4567.89,
            ["Boston", "Miami", "Denver"],
            {
                "best": 4567.89,
                "worst": 6000.0,
                "average": 5234.56,
                "total_simulations": 100,
            },
        )

        result = cli_runner.invoke(cli, ["simulate"])

        assert result.exit_code == 0
        assert "RESULTS" in result.output
        assert "4567.89 km" in result.output
        assert "Boston" in result.output


class TestCitiesCLI:
    """Test suite for the cities CLI command."""

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_cities_command_basic(self, mock_tsp, mock_create_df, cli_runner):
        """Test cities command with basic city list."""
        mock_df = pd.DataFrame(
            {
                "city": ["Boston", "Miami"],
                "latitude": [42.3601, 25.7617],
                "longitude": [-71.0589, -80.1918],
            }
        )
        mock_create_df.return_value = mock_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami"])

        result = cli_runner.invoke(cli, ["cities", "Boston", "Miami"])

        assert result.exit_code == 0
        assert "Creating route for 2 cities" in result.output
        assert "Boston" in result.output
        assert "Miami" in result.output

    def test_cities_command_insufficient_cities(self, cli_runner):
        """Test cities command with less than 2 cities."""
        result = cli_runner.invoke(cli, ["cities", "Boston"])

        assert result.exit_code == 0
        assert "Please provide at least 2 cities" in result.output

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_cities_command_custom_simulations(
        self, mock_tsp, mock_create_df, cli_runner
    ):
        """Test cities command with custom number of simulations."""
        mock_df = pd.DataFrame(
            {
                "city": ["Boston", "Miami"],
                "latitude": [42.3601, 25.7617],
                "longitude": [-71.0589, -80.1918],
            }
        )
        mock_create_df.return_value = mock_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami"])

        result = cli_runner.invoke(
            cli, ["cities", "Boston", "Miami", "--n_simulations", "500"]
        )

        assert result.exit_code == 0
        assert "Running 500 simulations" in result.output

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_cities_command_with_output(self, mock_tsp, mock_create_df, cli_runner):
        """Test cities command with output file."""
        mock_df = pd.DataFrame(
            {
                "city": ["Boston", "Miami"],
                "latitude": [42.3601, 25.7617],
                "longitude": [-71.0589, -80.1918],
            }
        )
        mock_create_df.return_value = mock_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami"])

        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(
                cli, ["cities", "Boston", "Miami", "--output", "custom.txt"]
            )

            assert result.exit_code == 0
            assert "Results saved to custom.txt" in result.output

            # Check file content
            with open("custom.txt", "r", encoding="utf8") as f:
                content = f.read()
                assert "Custom Cities" in content


# ============================================
# EDGE CASES AND ERROR HANDLING
# ============================================


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_tsp_with_single_city(self):
        """Test TSP with only one city (should still work)."""
        df = pd.DataFrame(
            {"city": ["Boston"], "latitude": [42.3601], "longitude": [-71.0589]}
        )

        distance, route = tsp(df)

        # Distance should be 0 (or very close) for single city
        assert distance < 1  # Less than 1 km
        assert len(route) == 1

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_with_single_simulation(
        self, mock_tsp, mock_create_df, sample_cities_df
    ):
        """Test main with n_simulations=1."""
        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        stats = main(1, verbose=False)[2]

        assert stats["total_simulations"] == 1
        assert stats["best"] == stats["worst"] == stats["average"]

    def test_cli_help_command(self, cli_runner):
        """Test that help command works."""
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "TSP Solver" in result.output
        assert "simulate" in result.output
        assert "cities" in result.output

    def test_simulate_help(self, cli_runner):
        """Test simulate command help."""
        result = cli_runner.invoke(cli, ["simulate", "--help"])

        assert result.exit_code == 0
        assert "Monte Carlo simulations" in result.output
        assert "--n_simulations" in result.output
        assert "--verbose" in result.output


# ============================================
# PARAMETRIZED TESTS
# ============================================


class TestParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("n_cities", [2, 5, 10, 15])
    def test_tsp_with_varying_city_counts(self, n_cities):
        """Test TSP with different numbers of cities."""
        cities = [f"City{i}" for i in range(n_cities)]
        latitudes = [40.0 + i for i in range(n_cities)]
        longitudes = [-70.0 - i for i in range(n_cities)]

        df = pd.DataFrame(
            {"city": cities, "latitude": latitudes, "longitude": longitudes}
        )

        distance, route = tsp(df)

        assert len(route) == n_cities
        assert distance > 0

    @pytest.mark.parametrize("n_sims", [1, 10, 100])
    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_with_varying_simulations(
        self, mock_tsp, mock_create_df, n_sims, sample_cities_df
    ):
        """Test main with different simulation counts."""
        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        _, _, stats = main(n_sims, verbose=False)

        assert stats["total_simulations"] == n_sims
        assert mock_tsp.call_count == n_sims


# ============================================
# PERFORMANCE TESTS
# ============================================


class TestPerformance:
    """Performance-related tests."""

    @patch("fetch_cities_lat_long.create_cities_dataframe")
    @patch("fetch_cities_lat_long.tsp")
    def test_main_completes_in_reasonable_time(
        self, mock_tsp, mock_create_df, sample_cities_df
    ):
        """Test that main completes in reasonable time."""
        import time

        mock_create_df.return_value = sample_cities_df
        mock_tsp.return_value = (5000.0, ["Boston", "Miami", "Denver"])

        start = time.time()
        main(100, verbose=False)
        elapsed = time.time() - start

        # With mocking, should complete very quickly
        assert elapsed < 1.0  # Less than 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
