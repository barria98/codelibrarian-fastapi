from fastapi import APIRouter

router = APIRouter(prefix="/users")

@router.get("/{id}")
def read_user(id: int):
    return {"id": id}
