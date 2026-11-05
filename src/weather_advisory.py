
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------------------------------
# 1. CREATE SESSION WITH AUTOMATIC RETRIES
# --------------------------------------------------

def create_session():
    """Create an HTTP session with automatic retries."""

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)

    return session


# --------------------------------------------------
# 2. GENERATE AGRICULTURAL WEATHER ADVISORIES
# --------------------------------------------------

def generate_agricultural_alerts(
    temperature,
    humidity,
    precipitation,
    wind_speed,
    rain_probability
):
    """Generate general agricultural advisories from weather data."""

    alerts = []

    # Temperature advisory
    if temperature >= 40:
        alerts.append(
            "CRITICAL HEAT: Temperature is extremely high. "
            "Avoid unnecessary field work during peak afternoon hours "
            "and monitor crops for heat stress."
        )

    elif temperature >= 35:
        alerts.append(
            "HIGH TEMPERATURE: Heat stress may affect crops. "
            "Monitor soil moisture and plan irrigation carefully."
        )

    elif temperature <= 5:
        alerts.append(
            "LOW TEMPERATURE: Cold conditions may affect sensitive crops. "
            "Monitor crops for cold or frost damage."
        )

    # Humidity advisory
    if humidity >= 85:
        alerts.append(
            "HIGH HUMIDITY: Moist conditions may increase the risk "
            "of fungal diseases. Monitor leaves and crop health."
        )

    elif humidity <= 30:
        alerts.append(
            "LOW HUMIDITY: Dry air may increase crop water loss. "
            "Monitor soil moisture and crop stress."
        )

    # Rainfall advisory
    if rain_probability is not None and rain_probability >= 80:
        alerts.append(
            "VERY HIGH RAIN CHANCE: Postpone irrigation if rainfall "
            "is sufficient. Avoid spraying before expected rain."
        )

    elif rain_probability is not None and rain_probability >= 60:
        alerts.append(
            "RAIN EXPECTED: Check field drainage and weather conditions "
            "before irrigation or pesticide application."
        )

    # Current precipitation
    if precipitation >= 10:
        alerts.append(
            "HEAVY CURRENT PRECIPITATION: Check fields for waterlogging "
            "and ensure proper drainage."
        )

    elif precipitation > 0:
        alerts.append(
            "RAINFALL DETECTED: Check soil moisture before scheduling "
            "additional irrigation."
        )

    # Wind advisory
    if wind_speed >= 40:
        alerts.append(
            "VERY STRONG WIND: Avoid pesticide spraying and monitor "
            "standing crops for possible wind damage."
        )

    elif wind_speed >= 25:
        alerts.append(
            "STRONG WIND: Avoid spraying during windy conditions "
            "to reduce spray drift."
        )

    # General advisory
    if not alerts:
        alerts.append(
            "WEATHER CONDITIONS: No major alert was detected using "
            "the selected weather thresholds. Continue routine crop monitoring."
        )

    return alerts


# --------------------------------------------------
# 3. FETCH WEATHER INFORMATION
# --------------------------------------------------

def get_weather(city):
    """Fetch weather information and agricultural advisories."""

    if not isinstance(city, str) or not city.strip():
        return {"error": "Please enter a valid city name."}

    city = city.strip()
    session = create_session()

    try:

        # ------------------------------------------
        # STEP 1: FIND CITY COORDINATES
        # ------------------------------------------

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = session.get(
            geo_url,
            params={
                "name": city,
                "count": 10,
                "language": "en",
                "format": "json"
            },
            timeout=(10, 30)
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()
        results = geo_data.get("results", [])

        if not results:
            return {
                "error": f"City '{city}' was not found. Please check the spelling."
            }

        # Prefer exact city-name matches
        exact_matches = [
            place for place in results
            if place.get("name", "").casefold() == city.casefold()
        ]

        candidates = exact_matches if exact_matches else results

        # Prefer Indian locations when multiple matches exist
        place = next(
            (
                p for p in candidates
                if p.get("country_code") == "IN"
            ),
            candidates[0]
        )

        latitude = place.get("latitude")
        longitude = place.get("longitude")

        if latitude is None or longitude is None:
            return {
                "error": "Could not find geographical coordinates for this city."
            }

        # ------------------------------------------
        # STEP 2: FETCH WEATHER FORECAST
        # ------------------------------------------

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = session.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,

                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "wind_speed_10m"
                ),

                "daily": (
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_probability_max"
                ),

                "forecast_days": 1,
                "timezone": "auto"
            },
            timeout=(10, 30)
        )

        weather_response.raise_for_status()

        data = weather_response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        # ------------------------------------------
        # STEP 3: EXTRACT WEATHER VALUES
        # ------------------------------------------

        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        precipitation = current.get("precipitation")
        wind_speed = current.get("wind_speed_10m")

        max_temp = daily.get("temperature_2m_max", [None])[0]
        min_temp = daily.get("temperature_2m_min", [None])[0]

        rain_probability = daily.get(
            "precipitation_probability_max",
            [None]
        )[0]

        required_values = [
            temperature,
            humidity,
            precipitation,
            wind_speed,
            max_temp,
            min_temp
        ]

        if any(value is None for value in required_values):
            return {
                "error": "Weather information is incomplete for this city."
            }

        # ------------------------------------------
        # STEP 4: GENERATE AGRICULTURAL ADVISORIES
        # ------------------------------------------

        alerts = generate_agricultural_alerts(
            temperature=temperature,
            humidity=humidity,
            precipitation=precipitation,
            wind_speed=wind_speed,
            rain_probability=rain_probability
        )

        # ------------------------------------------
        # STEP 5: RETURN WEATHER INFORMATION
        # ------------------------------------------

        return {
            "city": place.get("name", city),
            "country": place.get("country", ""),
            "region": place.get("admin1", ""),

            "latitude": latitude,
            "longitude": longitude,

            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "wind_speed": wind_speed,

            "max_temperature": max_temp,
            "min_temperature": min_temp,

            "rain_probability": rain_probability,

            "alerts": alerts
        }

    # ----------------------------------------------
    # STEP 6: ERROR HANDLING
    # ----------------------------------------------

    except requests.exceptions.Timeout:
        return {
            "error": "Weather API timed out after retries. Please try again."
        }

    except requests.exceptions.ConnectionError:
        return {
            "error": (
                "Could not connect to Open-Meteo. "
                "Please check your internet connection."
            )
        }

    except requests.exceptions.HTTPError as error:
        return {
            "error": f"Weather API returned an HTTP error: {error}"
        }

    except requests.exceptions.RequestException as error:
        return {
            "error": f"Weather request failed: {error}"
        }

    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {
            "error": f"Could not process weather information: {error}"
        }

    finally:
        session.close()
