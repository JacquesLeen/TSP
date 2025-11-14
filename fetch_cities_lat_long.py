#!/usr/bin/env python

"""
TSP Solver - Command Line Tool

A command-line tool that simulates travelling between cities to find the shortest
path for the Travelling Salesman Problem (TSP) using Monte Carlo simulations.

Usage:
    ./fetch_cities_lat_long.py simulate [OPTIONS]
    ./fetch_cities_lat_long.py cities [CITIES...] [OPTIONS]

Dependencies:
    - geopy: Geographic calculations
    - pandas: Data manipulation
    - click: CLI framework
"""
from random import shuffle
import geopy
import geopy.distance
import pandas as pd
import click


def cities_from_args(*args):
    """
    Convert variable arguments into a list of city names.

    This helper function takes any number of city name arguments and converts
    them into a list for processing by other functions.

    Parameters
    ----------
    *args : str
        Variable number of city name strings.

    Returns
    -------
    list of str
        A list containing all city names passed as arguments.

    Examples
    --------
    >>> cities = cities_from_args("Boston", "Miami", "Seattle")
    >>> print(cities)
    ['Boston', 'Miami', 'Seattle']

    >>> cities = cities_from_args("New York")
    >>> print(cities)
    ['New York']

    See Also
    --------
    create_cities_dataframe : Uses this function to process custom city lists
    """
    return list(args)


def create_cities_dataframe(cities=None):
    """
    Create a pandas DataFrame containing cities with their geographic coordinates.

    This function generates a DataFrame with city names and retrieves their
    latitude and longitude coordinates using the Nominatim geocoding service from geopy.

    If no cities are provided, it defaults to 10 major US cities: New York, Los Angeles,
    Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, and San Jose.

    Parameters
    ----------
    cities : list of str, optional
        List of city names to geocode. If None, uses default US cities.
        Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns:
        - 'city' (str): Name of the city
        - 'latitude' (float): Latitude coordinate in decimal degrees
        - 'longitude' (float): Longitude coordinate in decimal degrees

    Raises
    ------
    geopy.exc.GeocoderTimedOut
        If the geocoding service times out while retrieving coordinates.
    geopy.exc.GeocoderServiceError
        If the geocoding service is unavailable or returns an error.
    AttributeError
        If a city cannot be found and location.latitude/longitude is accessed on None.

    Notes
    -----
    - This function requires an active internet connection to query the Nominatim API.
    - Nominatim has usage limits; excessive requests may result in rate limiting.
    - Each geocoding request may take several seconds to complete.
    - Consider caching results to avoid repeated API calls during development.

    Examples
    --------
    >>> # Use default cities
    >>> df = create_cities_dataframe()
    >>> print(df.head())
              city   latitude   longitude
    0     New York  40.712776  -74.005974
    1  Los Angeles  34.052235 -118.243683
    2      Chicago  41.878113  -87.629799

    >>> # Use custom cities
    >>> custom_cities = ["Boston", "Miami", "Denver"]
    >>> df = create_cities_dataframe(cities=custom_cities)
    >>> print(df)
          city   latitude   longitude
    0   Boston  42.360081  -71.058884
    1    Miami  25.761681  -80.191788
    2   Denver  39.739236 -104.990251

    >>> # Access specific city coordinates
    >>> ny_coords = df[df['city'] == 'New York'][['latitude', 'longitude']].values[0]
    >>> print(f"New York: {ny_coords}")
    New York: [40.712776 -74.005974]

    See Also
    --------
    geopy.geocoders.Nominatim : The geocoding service used
    pandas.DataFrame : The return type of this function
    cities_from_args : Helper function for processing city arguments
    """
    # Use default cities if none provided
    if cities is None:
        cities = [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
            "Dallas",
            "San Jose",
        ]

    # Create lists for coordinates
    latitudes = []
    longitudes = []

    # Geocode each city
    for city in cities:
        geolocator = geopy.geocoders.Nominatim(user_agent="tsp_pandas")
        location = geolocator.geocode(city, timeout=None)
        latitudes.append(location.latitude)
        longitudes.append(location.longitude)

    # Create and return DataFrame
    df = pd.DataFrame({"city": cities, "latitude": latitudes, "longitude": longitudes})
    return df


def tsp(cities_df):
    """
    Calculate the total distance of a randomized Travelling Salesman Problem (TSP) route.

    This function generates a random route through all cities in the provided DataFrame
    and calculates the total round-trip distance in kilometers. The route forms a cycle,
    returning to the starting city after visiting all others.

    Parameters
    ----------
    cities_df : pd.DataFrame
        DataFrame containing city information with required columns:
        - 'city' (str): Name of each city
        - 'latitude' (float): Latitude coordinate in decimal degrees
        - 'longitude' (float): Longitude coordinate in decimal degrees

    Returns
    -------
    tuple of (float, list of str)
        A tuple containing:
        - total_distance : float
            The total round-trip distance in kilometers
        - city_list : list of str
            The randomized order of cities visited in the route

    Notes
    -----
    - The function uses random.shuffle() to randomize the city order, so results
      will vary between calls unless a random seed is set.
    - Distances are calculated using the geodesic distance (accounts for Earth's curvature).
    - The route automatically returns to the starting city to complete the cycle.
    - This is a random solution, not an optimized one. For optimization, consider
      algorithms like nearest neighbor, 2-opt, or simulated annealing.

    Side Effects
    ------------
    Prints the randomized city list to stdout.

    Examples
    --------
    >>> cities_df = create_cities_dataframe()
    >>> total_dist, route = tsp(cities_df)
    Randomized city list: ['Chicago', 'Houston', 'Phoenix', ...]
    >>> print(f"Total distance: {total_dist:.2f} km")
    Total distance: 12543.67 km
    >>> print(f"Route: {' → '.join(route)} → {route[0]}")
    Route: Chicago → Houston → Phoenix → ... → Chicago

    >>> # Run multiple times to compare random solutions
    >>> distances = [tsp(cities_df)[0] for _ in range(100)]
    >>> best_distance = min(distances)
    >>> print(f"Best random solution: {best_distance:.2f} km")
    Best random solution: 9847.32 km

    See Also
    --------
    geopy.distance.distance : The distance calculation function used
    random.shuffle : The randomization method used
    create_cities_dataframe : Function to create the input DataFrame
    main : Runs multiple TSP simulations to find the best route
    """
    # Get list of cities and randomize order
    city_list = cities_df["city"].to_list()
    shuffle(city_list)
    print(f"Randomized city list: {city_list}")

    distance_list = []

    # Calculate distances between consecutive cities
    for i in range(len(city_list)):
        # Get current and next city indices
        current_city = city_list[i]
        next_city = city_list[(i + 1) % len(city_list)]  # Wraps to first city at end

        # Calculate distance between cities
        distance = geopy.distance.distance(
            (
                cities_df.loc[lambda df: df["city"] == current_city]["latitude"].values[
                    0
                ],
                cities_df.loc[lambda df: df["city"] == current_city][
                    "longitude"
                ].values[0],
            ),
            (
                cities_df.loc[lambda df: df["city"] == next_city]["latitude"].values[0],
                cities_df.loc[lambda df: df["city"] == next_city]["longitude"].values[
                    0
                ],
            ),
        )

        distance_list.append(distance.km)

    # Calculate and return total distance
    total_distance = sum(distance_list)
    return total_distance, city_list


def main(n_simulations, verbose=False):
    """
    Run multiple TSP simulations and return the best route found.

    This function performs Monte Carlo simulations by running the TSP algorithm
    multiple times with random city orderings and tracking the best (shortest)
    route discovered.

    Parameters
    ----------
    n_simulations : int
        Number of random simulations to run. More simulations increase the
        probability of finding a better solution.
    verbose : bool, optional
        If True, prints detailed output for each simulation showing distances
        and when new best routes are found. Default is False.

    Returns
    -------
    tuple of (float, list of str, dict)
        A tuple containing:
        - best_distance : float
            The shortest distance found in kilometers
        - best_route : list of str
            The city order that produced the shortest distance
        - stats : dict
            Dictionary with keys:
            - 'best' : float - Best (shortest) distance found
            - 'worst' : float - Worst (longest) distance found
            - 'average' : float - Average distance across all simulations
            - 'total_simulations' : int - Number of simulations run

    Notes
    -----
    - Uses default US cities from create_cities_dataframe()
    - Each simulation generates a random route and calculates its distance
    - More simulations generally lead to better results but take longer
    - Results are not guaranteed to be optimal, just the best found

    Performance Guidelines
    ----------------------
    - 100 simulations: ~1-2 seconds (good for testing)
    - 1,000 simulations: ~10-20 seconds (recommended for 10 cities)
    - 10,000 simulations: ~1-2 minutes (better optimization)
    - 100,000+ simulations: Several minutes (diminishing returns)

    Examples
    --------
    >>> # Run 100 simulations without verbose output
    >>> best_dist, best_route, stats = main(100)
    >>> print(f"Best distance: {best_dist:.2f} km")
    Best distance: 9847.32 km
    >>> print(f"Average distance: {stats['average']:.2f} km")
    Average distance: 11234.56 km

    >>> # Run with verbose output
    >>> best_dist, best_route, stats = main(50, verbose=True)
    Simulation 1: 12543.67 km ← New best!
    Simulation 2: 11892.34 km ← New best!
    ...

    >>> # Access statistics
    >>> _, _, stats = main(1000)
    >>> improvement = ((stats['worst'] - stats['best']) / stats['worst']) * 100
    >>> print(f"Best route is {improvement:.1f}% better than worst")
    Best route is 28.5% better than worst

    See Also
    --------
    tsp : Single TSP simulation
    create_cities_dataframe : Creates the city data used
    simulate : CLI command that uses this function
    """
    # Create cities DataFrame
    cities_df = create_cities_dataframe()

    # Initialize tracking variables
    best_distance = float("inf")
    best_route = None
    all_distances = []

    # Run simulations
    for i in range(n_simulations):
        distance_km, route = tsp(cities_df)
        all_distances.append(distance_km)

        # Check if this is the best route so far
        if distance_km < best_distance:
            best_distance = distance_km
            best_route = route
            if verbose:
                click.echo(f"Simulation {i+1}: {distance_km:.2f} km ← New best!")
        elif verbose:
            click.echo(f"Simulation {i+1}: {distance_km:.2f} km")

    # Calculate statistics
    stats = {
        "best": best_distance,
        "worst": max(all_distances),
        "average": sum(all_distances) / len(all_distances),
        "total_simulations": n_simulations,
    }

    return best_distance, best_route, stats


# ============================================
# CLI COMMANDS
# ============================================


@click.group()
def cli():
    """
    TSP Solver - Command Line Tool

    A tool that calculates the shortest distance between cities using their
    geographic coordinates. Uses randomized Monte Carlo simulations to find
    near-optimal routes for the Travelling Salesman Problem (TSP).

    Commands:
        simulate    Run multiple random simulations (recommended)
        cities      Run simulation with custom cities

    Examples:
        ./fetch_cities_lat_long.py simulate --n_simulations 1000
        ./fetch_cities_lat_long.py cities Boston Miami Denver
    """


@cli.command("simulate")
@click.option(
    "--n_simulations",
    default=100,
    type=int,
    help="Number of random simulations to run (default: 100)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Optional: Save results to a file (e.g., results.txt)",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print detailed output for each simulation",
)
def simulate(n_simulations, output, verbose):
    """
    Run Monte Carlo simulations to find the shortest TSP route.

    This command performs multiple random simulations of the Travelling Salesman
    Problem, generating different route permutations and tracking the shortest
    distance found. The more simulations you run, the higher the probability
    of finding a near-optimal solution.

    Uses default US cities: New York, Los Angeles, Chicago, Houston, Phoenix,
    Philadelphia, San Antonio, San Diego, Dallas, and San Jose.

    \b
    Algorithm:
    ----------
    1. Load city coordinates using Nominatim geocoding service
    2. For each simulation:
       - Randomly shuffle the city order
       - Calculate total round-trip distance
       - Track if this is the best route found
    3. Return the shortest route and distance with statistics

    \b
    Performance Notes:
    ------------------
    - 100 simulations: ~1-2 seconds (good for testing)
    - 1,000 simulations: ~10-20 seconds (recommended for 10 cities)
    - 10,000 simulations: ~1-2 minutes (better optimization)
    - 100,000+ simulations: Several minutes (diminishing returns)

    \b
    Examples:
    ---------
    Run with default settings (100 simulations):
        $ ./fetch_cities_lat_long.py simulate

    Run 1000 simulations for better results:
        $ ./fetch_cities_lat_long.py simulate --n_simulations 1000

    Run with verbose output:
        $ ./fetch_cities_lat_long.py simulate --n_simulations 500 --verbose

    Save results to a file:
        $ ./fetch_cities_lat_long.py simulate --n_simulations 5000 --output results.txt

    \b
    Output Format:
    --------------
    The command displays:
    - Progress indicator
    - Best route found (city order)
    - Shortest distance (in kilometers)
    - Statistics (if available)

    \b
    See Also:
    ---------
    cities : Run simulation with custom city list

    For deterministic optimization algorithms, consider implementing:
    - Nearest Neighbor algorithm
    - 2-opt optimization
    - Genetic algorithms
    - Simulated annealing
    """
    click.echo(f"Starting {n_simulations} TSP simulations...")
    click.echo("=" * 60)

    # Run simulations
    best_distance, best_route, stats = main(n_simulations, verbose=verbose)

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("RESULTS")
    click.echo("=" * 60)
    click.echo(f"Simulations run: {stats['total_simulations']}")
    click.echo(f"Best distance: {best_distance:.2f} km")
    click.echo(f"Average distance: {stats['average']:.2f} km")
    click.echo(f"Worst distance: {stats['worst']:.2f} km")
    click.echo(f"\nBest route: {' → '.join(best_route)} → {best_route[0]}")
    click.echo("=" * 60)

    # Save to file if requested
    if output:
        with open(output, "w", encoding="utf8") as f:
            f.write("TSP SIMULATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total simulations: {stats['total_simulations']}\n")
            f.write(f"Best distance: {best_distance:.2f} km\n")
            f.write(f"Average distance: {stats['average']:.2f} km\n")
            f.write(f"Worst distance: {stats['worst']:.2f} km\n")
            f.write("\nBest route:\n")
            for i, city in enumerate(best_route, 1):
                f.write(f"  {i}. {city}\n")
            f.write(f"  → Back to {best_route[0]}\n")
        click.echo(f"\n✓ Results saved to {output}")

    click.echo("\n✓ Simulation complete!")


@cli.command("cities")
@click.argument("cities", nargs=-1, required=True)
@click.option(
    "--n_simulations",
    default=100,
    type=int,
    help="Number of random simulations to run (default: 100)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Optional: Save results to a file",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print detailed output for each simulation",
)
def cities_command(cities, n_simulations, output, verbose):
    """
    Run TSP simulations with a custom list of cities.

    This command allows you to specify your own list of cities instead of using
    the default US cities. Provide city names as space-separated arguments.

    The command will:\n
    1. Geocode each city to get coordinates \n
    2. Run the specified number of simulations \n
    3. Display the best route found

    \b
    Examples:
    ---------
    Run with three cities:
        $ ./fetch_cities_lat_long.py cities Boston Miami Denver

    Run with more simulations:
        $ ./fetch_cities_lat_long.py cities Boston Miami Denver Seattle --n_simulations 1000

    Use cities with multiple words (use quotes):
        $ ./fetch_cities_lat_long.py cities "New York" "Los Angeles" "San Francisco"

    Save results to file:
        $ ./fetch_cities_lat_long.py cities Boston Miami --output my_results.txt

    \b
    Notes:
    ------
    - City names must be recognizable by the Nominatim geocoding service
    - For multi-word city names, use quotes: "New York"
    - Requires active internet connection for geocoding
    - More cities = longer computation time

    \b
    Arguments:
    ----------
    CITIES : Text arguments (required)
        Names of cities to include in the route. Provide as space-separated values.
    """
    if len(cities) < 2:
        click.echo("Error: Please provide at least 2 cities", err=True)
        return

    click.echo(f"Creating route for {len(cities)} cities: {', '.join(cities)}")
    click.echo("Fetching coordinates...")
    click.echo("=" * 60)

    # Create DataFrame with custom cities
    try:
        city_list = list(cities)
        cities_df = create_cities_dataframe(cities=city_list)
    except Exception as e:
        click.echo(f"Error geocoding cities: {e}", err=True)
        return

    # Run simulations
    click.echo(f"\nRunning {n_simulations} simulations...")

    best_distance = float("inf")
    best_route = None
    all_distances = []

    for i in range(n_simulations):
        distance_km, route = tsp(cities_df)
        all_distances.append(distance_km)

        if distance_km < best_distance:
            best_distance = distance_km
            best_route = route
            if verbose:
                click.echo(f"Simulation {i+1}: {distance_km:.2f} km ← New best!")
        elif verbose:
            click.echo(f"Simulation {i+1}: {distance_km:.2f} km")

    # Calculate statistics
    stats = {
        "best": best_distance,
        "worst": max(all_distances),
        "average": sum(all_distances) / len(all_distances),
        "total_simulations": n_simulations,
    }

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("RESULTS")
    click.echo("=" * 60)
    click.echo(f"Cities: {len(cities)}")
    click.echo(f"Simulations run: {stats['total_simulations']}")
    click.echo(f"Best distance: {best_distance:.2f} km")
    click.echo(f"Average distance: {stats['average']:.2f} km")
    click.echo(f"Worst distance: {stats['worst']:.2f} km")
    click.echo(f"\nBest route: {' → '.join(best_route)} → {best_route[0]}")
    click.echo("=" * 60)

    # Save to file if requested
    if output:
        with open(output, "w", encoding="utf8") as f:
            f.write("TSP SIMULATION RESULTS (Custom Cities)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Cities: {', '.join(cities)}\n")
            f.write(f"Total simulations: {stats['total_simulations']}\n")
            f.write(f"Best distance: {best_distance:.2f} km\n")
            f.write(f"Average distance: {stats['average']:.2f} km\n")
            f.write(f"Worst distance: {stats['worst']:.2f} km\n")
            f.write("\nBest route:\n")
            for i, city in enumerate(best_route, 1):
                f.write(f"  {i}. {city}\n")
            f.write(f"  → Back to {best_route[0]}\n")
        click.echo(f"\n✓ Results saved to {output}")

    click.echo("\n✓ Simulation complete!")


if __name__ == "__main__":
    cli()
