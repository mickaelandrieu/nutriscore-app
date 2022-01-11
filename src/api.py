"""Basic REST API to expose the predictions of the ML app."""
import joblib
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

import config

app = FastAPI()
model = joblib.load("{0}best_model.z".format(config.MODELS))


class Product(BaseModel):
    """Product expected model from REST API request."""

    energy: float
    saturated_fat: float
    sugars: float
    salt: float
    fibers: float
    pnns_group: str


@app.post("/")
def predict(product: Product):
    """Return ML predictions, see /docs for more information.

    Args:
        product: (Product) the parsed data from user request

    Returns:
        A dictionnary with the predicted nutrigrade
        and the related probability
    """
    sample = {
        "energy": round(float(product.energy)),
        "saturated_fat": round(float(product.saturated_fat)),
        "sugars": round(float(product.sugars)),
        "salt": round(float(product.salt)),
        "fibers": round(float(product.fibers)),
        "group1_Beverages": 0,
        "group1_Cereals and potatoes": 0,
        "group1_Composite foods": 0,
        "group1_Fat and sauces": 0,
        "group1_Fruits and vegetables": 0,
        "group1_Milk and dairy products": 0,
        "group1_Sugary snacks": 0,
        "group1_unknown": 0,
    }

    # If category is detected then assign the property value to 1.
    formatted_category = "group1_{0}".format(product.pnns_group)

    if formatted_category in sample.keys():
        sample[formatted_category] = 1

    sample = list(sample.values())

    # Predict the nutrigrade !
    nutrigrade = model.predict([sample])[0]
    probability = model.predict_proba([sample]).argmax(1).item()

    # Return of the prediction and the probability
    return {"nutrigrade": nutrigrade, "probability": probability}


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.DOCKER_HOST,
        port=config.DOCKER_PORT,
        reload=True,
        debug=True,
        workers=3,
    )
