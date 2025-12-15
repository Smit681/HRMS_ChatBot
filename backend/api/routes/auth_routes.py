from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pymongo import MongoClient
from datetime import timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
from config import Config

from ..auth_schemas import UserRegister, UserLogin, Token, UserResponse
from ..auth_utils import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

# MongoDB setup
client = MongoClient(Config.MONGODB_URI)
db = client[Config.MONGODB_DB_NAME]
users_collection = db['users']

# Create unique index on email
users_collection.create_index("email", unique=True)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency to get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    user = users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """Register a new user"""
    
    # Check if user already exists
    if users_collection.find_one({"email": user_data.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user document
    from datetime import datetime
    user_doc = {
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": get_password_hash(user_data.password),
        "created_at": datetime.utcnow().isoformat(),
        "is_active": True
    }
    
    # Insert into database
    result = users_collection.insert_one(user_doc)
    
    return UserResponse(
        email=user_doc["email"],
        full_name=user_doc["full_name"],
        created_at=user_doc["created_at"]
    )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    
    # Find user
    user = users_collection.find_one({"email": form_data.username})
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information"""
    return UserResponse(
        email=current_user["email"],
        full_name=current_user["full_name"],
        created_at=current_user["created_at"]
    )