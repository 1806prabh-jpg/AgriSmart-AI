
import pandas as pd


def get_sample_market_data():
    """
    Demonstration market prices.
    These values are illustrative, NOT live mandi prices.
    Prices are in INR per quintal.
    """

    data = [
        ["Wheat", "Ludhiana", 2300, 2600, 2500],
        ["Wheat", "Jalandhar", 2250, 2550, 2450],
        ["Wheat", "Amritsar", 2350, 2650, 2550],
        ["Wheat", "Phagwara", 2280, 2580, 2480],

        ["Rice", "Ludhiana", 2800, 3200, 3000],
        ["Rice", "Jalandhar", 2750, 3150, 2950],
        ["Rice", "Amritsar", 2850, 3250, 3100],
        ["Rice", "Phagwara", 2780, 3180, 2980],

        ["Maize", "Ludhiana", 1800, 2200, 2000],
        ["Maize", "Jalandhar", 1850, 2250, 2100],
        ["Maize", "Amritsar", 1750, 2150, 1950],
        ["Maize", "Phagwara", 1820, 2220, 2050],

        ["Cotton", "Ludhiana", 6500, 7200, 6900],
        ["Cotton", "Jalandhar", 6400, 7100, 6800],
        ["Cotton", "Amritsar", 6600, 7300, 7000],
        ["Cotton", "Phagwara", 6450, 7150, 6850],
    ]

    columns = [
        "Crop",
        "Mandi",
        "Min Price",
        "Max Price",
        "Modal Price"
    ]

    return pd.DataFrame(data, columns=columns)


def analyze_market_data(df, selected_crop):
    """
    Filter market data for the selected crop
    and calculate price statistics.
    """

    crop_data = df[
        df["Crop"].str.lower() == selected_crop.lower()
    ].copy()

    if crop_data.empty:
        return None

    average_price = crop_data["Modal Price"].mean()
    highest_price = crop_data["Modal Price"].max()
    lowest_price = crop_data["Modal Price"].min()

    best_mandi = crop_data.loc[
        crop_data["Modal Price"].idxmax(), "Mandi"
    ]

    return {
        "data": crop_data,
        "average_price": average_price,
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "best_mandi": best_mandi
    }
