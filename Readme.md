# BentoML
## Steps:
### Build & Train Model
* downloaded small sample dataset
* train_model.py > generate model&scaler pkl files

### Bentoml service
* sample Wine service with one endpoint > service.py
* try and serve locally to check everything is fine
> bentoml serve service:Wine 
* bentofile.yaml with required pip installs
* build bentoml
> bentoml build
* containerize (on mac so i need to specify platform)
> bentoml containerize --platform=linux/amd64 wine:latest
* image will be saved to the local registery
* run docker image locally with port 3000 exposed
> docker run --rm -p 3000:3000 wine:latest