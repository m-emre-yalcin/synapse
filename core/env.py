from dotenv import load_dotenv
import os


def init_env():
    load_dotenv()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
