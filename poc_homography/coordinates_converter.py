# coordinates_converter.py


class CoordinatesConverter:
    """
    Utility class to convert Degrees, Minutes, Seconds (DMS)
    geographic coordinates to Decimal Degrees (DD).
    """

    @staticmethod
    def dms_to_dd(dms_str: str) -> float:
        """
        Converts a DMS string (e.g., "41°19'46.8\"N") to Decimal Degrees.
        """
        import re

        # Regex to capture Degrees, Minutes, Seconds, and Hemisphere
        match = re.match(r"(\d+)°(\d+)'([\d\.]+)\"([NSEW])", dms_str)

        if not match:
            raise ValueError(f"Invalid DMS format: {dms_str}")

        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        hemisphere = match.group(4)

        decimal_degrees = degrees + minutes / 60.0 + seconds / 3600.0

        # Apply negative sign for South and West
        if hemisphere in ("S", "W"):
            decimal_degrees *= -1

        return decimal_degrees
