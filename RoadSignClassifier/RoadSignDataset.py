import tensorflow as tf
import os.path


class RoadSignDataset:
    def __init__(self, datasetDir, imageWidth, imageHeight, batchSize):
        self.__DatasetDir = datasetDir
        self.__ImageWidth = imageWidth
        self.__ImageHeight = imageHeight
        self.__BatchSize = batchSize
        self.__autotune = tf.data.AUTOTUNE

        self.__LearnData = self.__loadData()

    def __loadData(self):
        return tf.keras.utils.image_dataset_from_directory(
            directory=self.__DatasetDir,
            shuffle=True,
            batch_size=None,
            image_size=(self.__ImageHeight, self.__ImageWidth),
        )

    def getClasses(self):
        return self.__LearnData.class_names

    def split(self, dataset, trainRatio, validRatio):
        learnDataLength = len(self.__LearnData)
        trainSize = int(learnDataLength * trainRatio)
        validSize = int(learnDataLength * validRatio)
        testSize = learnDataLength - trainSize - validSize

        trainData = self.__LearnData.take(trainSize)
        validData = self.__LearnData.skip(trainSize).take(validSize)
        testData = self.__LearnData.skip(trainSize + validSize).take(testSize)
        return (
            trainData.batch(self.__BatchSize),
            validData.batch(self.__BatchSize),
            testData.batch(self.__BatchSize),
        )

    def getData(self, trainRatio=0.7, validRatio=0.2):
        # Do buffered prefetching to avoid i/o blocking
        trainData, validData, testData = self.split(
            self.__LearnData, trainRatio, validRatio
        )

        trainData = trainData.prefetch(buffer_size=self.__autotune)
        testData = testData.prefetch(buffer_size=self.__autotune)
        validData = validData.prefetch(buffer_size=self.__autotune)

        return trainData, validData, testData

    def __createExample(self, image, label):

        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.io.encode_jpeg(image)

        feature = {
            "images": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image.numpy()])
            ),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def __parseRecord(self, example):

        feature_description = {
            "images": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["images"] = tf.image.convert_image_dtype(
            tf.io.decode_jpeg(example["images"], channels=3), dtype=tf.float32
        )

        return example["images"], example["labels"]

    def save2Records(self, path):
        if not os.path.isfile(path):
            with tf.io.TFRecordWriter(path) as writer:
                for image, label in self.__LearnData:
                    tf_example = self.__createExample(image, label)
                    writer.write(tf_example.SerializeToString())
        else:
            print("File already exists")

    def getDataFromRecords(self, path, trainRatio=0.7, validRatio=0.2):
        if not os.path.isfile(path):
            print("File does not exist")
            raise OSError("File does not exist")

        # Create a dataset from the TFRecord file.
        rawDataset = tf.data.TFRecordDataset(path)
        dataset = rawDataset.map(self.__parseRecord)

        trainData, validData, testData = self.split(dataset, trainRatio, validRatio)

        # Do buffered prefetching to avoid i/o blocking
        trainData = trainData.prefetch(buffer_size=self.__autotune)
        testData = testData.prefetch(buffer_size=self.__autotune)
        validData = validData.prefetch(buffer_size=self.__autotune)

        return trainData, validData, testData
