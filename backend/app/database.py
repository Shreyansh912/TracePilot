"""
backend/app/database.py
Asynchronous MongoDB connection lifecycle management via Motor.
"""
from motor.motor_asyncio import AsyncIOMotorClient
from backend.app.config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_to_mongo():
    global client, db
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
        db = client[settings.DATABASE_NAME]
        # Quick server ping check
        await client.admin.command('ping')
        print(f"[+] Successfully connected to MongoDB database: {settings.DATABASE_NAME}")

        # Ensure index on sessions and projects for high-speed queries
        await db["debug_sessions"].create_index("created_at")
        await db["debug_sessions"].create_index("project_id")
        await db["users"].create_index("email", unique=True)
    except Exception as e:
        print(f"[!] Warning: Could not connect to MongoDB at {settings.MONGODB_URI}: {e}")
        print("    Running in offline database mode until MongoDB is active.")


async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("[*] MongoDB connection pool closed.")


def get_database():
    return db