import typing as t
from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Feedback(BaseModel):
    prompt: str
    improvement: str


class TreeNode(BaseModel):
    children: t.List["TreeNode"]
    conversation: t.List[Message]
    feedback: t.Optional[Feedback]
    # Multiple random document orders
    responses: t.Optional[t.List[str]] 
    on_topic: t.Optional[bool]
    score: t.Optional[float]


class Parameters(BaseModel):
    model: str
    temperature: t.Optional[float] = None
    max_tokens: t.Optional[int] = None
    top_p: t.Optional[float] = None


ChatFunction = t.Callable[[t.List[Message]], Message]
Conversation = t.List[Message]


class Product(BaseModel):
    category: str
    brand: str
    model: str
    
    def __hash__(self):
        return hash((self.category, self.brand, self.model))
    
    def __eq__(self, other):
        return (
            (self.category, self.brand, self.model) ==
            (other.category, other.brand, other.model)
        )