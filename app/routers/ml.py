from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse

from app.service import model_select
from ml.linear import LinearBM

router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    default_response_class=ORJSONResponse,
    responses={404: {"description": "Not found"}},
)


@router.post("/")
async def training(train_params: LinearBM.TrainParams) -> ORJSONResponse:
    model = model_select.model_select(train_params.mission, train_params.model)
    # 使用模型进行训练
    model.train(train_params)
    # model.save()
    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "Status": "Success",
            "Mission": f"{train_params.mission}",
            "Model": f"{train_params.model}",
        },
    )
