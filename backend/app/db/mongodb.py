import motor.motor_asyncio
from app.core.config import settings

class DataBase:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db: motor.motor_asyncio.AsyncIOMotorDatabase = None

db = DataBase()

async def connect_to_mongo():
    print("Connecting to MongoDB...")
    db.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_CONNECTION_STRING)
    db.db = db.client[settings.MONGODB_DATABASE_NAME]
    print(f"Connected to MongoDB database: {settings.MONGODB_DATABASE_NAME}")

async def close_mongo_connection():
    print("Closing MongoDB connection...")
    db.client.close()
    print("MongoDB connection closed.")

def get_database() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    if db.db is None:
        # This situation should ideally be handled by ensuring connect_to_mongo is called at startup.
        # For now, let's raise an error or log, as direct synchronous connection here is not ideal with motor.
        raise RuntimeError("Database not initialized. Call connect_to_mongo() at application startup.")
    return db.db

# It's also common to have a function to get specific collections, e.g.:
# def get_collection(collection_name: str) -> motor.motor_asyncio.AsyncIOMotorCollection:
#     return get_database()[collection_name]
