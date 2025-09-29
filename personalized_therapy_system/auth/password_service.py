from passlib.context import CryptContext
from typing import Optional
import logging

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure bcrypt hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12 
)

def hash_password(password: str) -> Optional[str]:
    """
    Hash a plain text password using bcrypt.

    Args:
        password (str): The plain text password.

    Returns:
        Optional[str]: The hashed password, or None if an error occurs.
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against the hashed version.

    Args:
        plain_password (str): Raw password entered by user.
        hashed_password (str): Hashed password stored in DB.

    Returns:
        bool: True if match, False otherwise.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.warning(f"Password verification failed: {str(e)}")
        return False
