import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    datefmt="%Y-%m-%dT%H:%M:%S", # ISO 8601 timestamp format
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)


# Create a logger instance
def get_logger() -> logging.Logger:
    return logging.getLogger("Mintegrity")
