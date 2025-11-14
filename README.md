# 🗺️ TSP Solver - Travelling Salesman Problem CLI

A command-line tool that solves the Travelling Salesman Problem (TSP) using Monte Carlo simulations and real geographic coordinates from cities around the world.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Simulate Command](#simulate-command)
  - [Cities Command](#cities-command)
- [Examples](#-examples)
- [Performance](#-performance)
- [License](#-license)


## 🎯 Overview

The Travelling Salesman Problem (TSP) is a classic algorithmic problem in computer science and operations research. Given a list of cities and the distances between them, the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

This tool uses:
- **Monte Carlo simulations** to explore different route permutations
- **Real geographic coordinates** from the Nominatim geocoding service
- **Geodesic distance calculations** that account for Earth's curvature
- **Interactive CLI** built with Click framework

## ✨ Features

- 🌍 **Real-world cities** - Uses actual geographic coordinates via geopy
- 🎲 **Monte Carlo optimization** - Runs multiple random simulations to find near-optimal routes
- 🗺️ **Accurate distances** - Calculates geodesic distances in kilometers
- 🎨 **Beautiful CLI** - Interactive command-line interface with colored output
- 📊 **Statistics** - Provides best, worst, and average distances
- 💾 **Export results** - Save route results to files
- 🔧 **Customizable** - Use default US cities or specify your own

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/JacquesLeen/TSP.git
cd TSP

# Install dependencies
make install

# Make the script executable
chmod +x fetch_cities_lat_long.py
```
## 🚀 Quick Start

```bash
# Run with default settings (100 simulations, 10 US cities)
./fetch_cities_lat_long.py simulate

# Run with more simulations for better results
./fetch_cities_lat_long.py simulate --n_simulations 1000

# Use custom cities
./fetch_cities_lat_long.py cities Boston Miami Denver Seattle
```

## 📖 Usage

### Simulate Command

```bash
./fetch_cities_lat_long.py simulate [OPTIONS]
```

| Option | 	Type | Default | Description |
| ---- | 	---- | ---- | --- |
|--n_simulations| INTEGER | 100	| Number of random simulations to run|
|--output| PATH |	None  |	Save results to a file |
|--verbose | FLAG	| False | 	Show detailed output for each simulation|
|--help | FLAG |	-	| Show help message |


```bash
# Run with default settings (100 simulations, 10 US cities)
./fetch_cities_lat_long.py simulate

# Run with more simulations for better results
./fetch_cities_lat_long.py simulate --n_simulations 1000

# Use custom cities
./fetch_cities_lat_long.py cities Boston Miami Denver Seattle
```

### Cities Command
```bash
./fetch_cities_lat_long.py cities [CITIES...] [OPTIONS]
```

| Option | 	Type | Default | Description |
| ---- | 	---- | ---- | --- |
|--n_simulations| INTEGER | 100	| Number of random simulations to run|
|--output| PATH |	None  |	Save results to a file |
|--verbose | FLAG	| False | 	Show detailed output for each simulation|
|--help | FLAG |	-	| Show help message |

```bash
# Three cities
./fetch_cities_lat_long.py cities Boston Miami Denver

# Cities with multiple words (use quotes)
./fetch_cities_lat_long.py cities "New York" "Los Angeles" "San Francisco"

# With more simulations
./fetch_cities_lat_long.py cities Boston Miami Denver --n_simulations 1000

# Save results
./fetch_cities_lat_long.py cities Boston Miami --output my_route.txt

# European cities
./fetch_cities_lat_long.py cities Paris London Berlin Rome Madrid
```

## 💡 Examples

### Standard

```bash
$ ./fetch_cities_lat_long.py simulate


Starting 100 TSP simulations...
============================================================
Randomized city list: ['Chicago', 'Houston', 'Phoenix', ...]
============================================================
RESULTS
============================================================
Simulations run: 100
Best distance: 9847.32 km
Average distance: 11234.56 km
Worst distance: 13542.78 km

Best route: Chicago → Houston → Phoenix → San Diego → Los Angeles → San Jose → San Francisco → New York → Philadelphia → Dallas → Chicago
============================================================

✓ Simulation complete!
```
### Verbose

```
$ ./fetch_cities_lat_long.py simulate --n_simulations 50 --verbose

Starting 50 TSP simulations...
============================================================
Randomized city list: ['Chicago', 'Houston', 'Phoenix', ...]
Simulation 1: 12543.67 km ← New best!
Simulation 2: 11892.34 km ← New best!
Simulation 3: 13201.45 km
Simulation 4: 11234.56 km ← New best!
...
```
### Custom Cities

```
$ ./fetch_cities_lat_long.py cities Boston Miami Denver Seattle --n_simulations 500

Creating route for 4 cities: Boston, Miami, Denver, Seattle
Fetching coordinates...
============================================================

Running 500 simulations...
============================================================
RESULTS
============================================================
Cities: 4
Simulations run: 500
Best distance: 7845.23 km
Average distance: 8234.56 km
Worst distance: 9123.45 km

Best route: Boston → Miami → Denver → Seattle → Boston
============================================================

✓ Simulation complete!
```

## ⚡ Performance

```bash
# examples
time ./fetch_cities_lat_long.py simulate --n_simulations 100
time ./fetch_cities_lat_long.py simulate --n_simulations 1000
time ./fetch_cities_lat_long.py simulate --n_simulations 10000
```

## 📄 License

```
MIT License

Copyright (c) 2025 Giacomo Lini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
