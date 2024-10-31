import tensorflow as tf


class RoadSignDataset:
    def __init__(self, DatasetConfig):
        self.__DatasetDir = DatasetConfig["DIR"]
        self.__ImageWidth = DatasetConfig["IMAGE_WIDTH"]
        self.__ImageHeight = DatasetConfig["IMAGE_HEIGHT"]
        self.__BatchSize = DatasetConfig["BATCH_SIZE"]

    def __loadData(self):
        self.__LearnData = tf.keras.utils.image_dataset_from_directory(
            directory=self.__DatasetDir,
            shuffle=True,
            batch_size=self.__BatchSize,
            image_size=(self.__ImageHeight, self.__ImageWidth),
        )

    def getData(self, trainRatio=0.7, validationRatio=0.2):
        self.__loadData()
        # Split learn data into training, validation, and test sets
        trainSize = int(len(self.__LearnData) * trainRatio)
        validSize = int(len(self.__LearnData) * validationRatio)
        testSize = len(self.__LearnData) - trainSize - validSize

        trainData = self.__LearnData.take(trainSize)
        validData = self.__LearnData.skip(trainSize).take(validSize)
        testData = self.__LearnData.skip(trainSize + validSize).take(testSize)

        return trainData, validData, testData

    def getClasses(self):
        return self.__LearnData.class_names
