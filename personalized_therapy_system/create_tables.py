# create_tables.py

import asyncio

from database.db_config import Base, engine
from database import models 

async def create_all_tables():
    async with engine.begin() as conn:
        print("creating all tables...")
        await conn.run_sync(Base.metadata.create_all)
        print("Done!")

if __name__ == "__main__":
    asyncio.run(create_all_tables())
