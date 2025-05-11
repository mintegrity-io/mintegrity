from scripts.commons.logging_config import get_logger
log = get_logger()


def check_ethereum_address_validity(address: str) -> None:
    """
    Validates if the given string is a valid Ethereum address.

    :param address: The Ethereum address to validate
    :return: True if valid, False otherwise
    """
    if not address.startswith("0x") or len(address) != 42:
        raise ValueError(f"Invalid Ethereum address: {address}")
    else:
        log.debug(f"Valid Ethereum address: {address}")