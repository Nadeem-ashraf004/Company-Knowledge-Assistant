from datetime import datetime
from uuid import UUID ,uuid4

from pydantic import(
     BaseModel, 
     ConfigDict, 
     EmailStr, 
     Field ,
     field_validator , 
     model_validator
)


class UserCreate(BaseModel):
    full_name: str = Field(min_length=2 , max_length=100)
    email: EmailStr
    password: str = Field(
        min_length=8, 
        max_length=128
        )
    confrim_password : str =Field(
        min_length=8, 
        max_length=128
        )
    @model_validator(mode="after")
    
    def validate_full_name(self):
        self.full_name = " ".join(self.full_name.strip().split())

        if not self.full_name:
            raise ValueError(
                " full name is not emty"
                )
        if not any(char.isupper() for char in self.password):
            raise ValueError(
                "password must contain atleast one uppercase letter.. "
                )
        if not any(char.islower() for char in self.password):
            raise ValueError(
                "password must contain atleast one lowercase letter ..."
                )
        if not any(char.isdigit() for char in self.password):
            raise ValueError(
                "password must have atleast one digit..."
                )
        if self.password != self.confrim_password:
            raise ValueError(
                "password and confirm are not match"
                )
        return self
        


class UserLogin(BaseModel): 
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id : UUID
    full_name : str
    email : EmailStr
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"