from RoadSignClassifier import RoadSignDataset, RoadSignModel
import Configuration
import numpy as np


def main():
    appConfig = Configuration.config["app"]
    modelConfig = Configuration.config["model"]
    datasetConfig = Configuration.config["dataset"]

    model = RoadSignModel.RoadSignModel(
        inputWidth=datasetConfig["IMAGE_HEIGHT"],
        inputHeight=datasetConfig["IMAGE_HEIGHT"],
    )

    roadSignDataset = RoadSignDataset.RoadSignDataset(
        datasetDir=datasetConfig["DIR"],
        imageWidth=datasetConfig["IMAGE_WIDTH"],
        imageHeight=datasetConfig["IMAGE_HEIGHT"],
        batchSize=datasetConfig["BATCH_SIZE"],
    )
    trainData, validData, testData = roadSignDataset.getData()

    batchSize = datasetConfig["BATCH_SIZE"]
    print(
        "Train data size={0}, test data size={1} and validation data size={2}".format(
            len(list(trainData)) * batchSize,
            len(list(testData)) * batchSize,
            len(list(validData)) * batchSize,
        )
    )
    # Get class names from dataset
    classNames = roadSignDataset.getClasses()
    print(classNames)


if __name__ == "__main__":
    main()
